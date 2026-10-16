"""SIM1RGBTLoader — RGBT data loader for the SIM1 dataset.

Reads RGB face video (avi1) from Structured Study Data (extracted zips) and
thermal (.dat) from Raw Thermal Data, aligns them by fps-ratio, and produces
4-channel (R,G,B,T) clips with multi-task labels (HR, BR, EDA).

Expected directory layout:
  <data_path>/
    Raw Thermal Data/
      T001/  T001-BL.dat  T001-BL.inf  ...
    R-Friendly Study Data/
      T001.csv  ...
    Structured Study Data/
      extracted/          <- unzipped from *.zip
        T001/
          1 BL/  T001-001.avi1.avi  ...
          2 PD/  T001-002.avi1.avi  ...
        T002/
          ...

Alignment strategy:
  RGB is the master clock (27 fps).  For each RGB frame i, the thermal frame
  index is j = clip(round(i * T_fps / RGB_fps), 0, n_thermal-1) — nearest-
  frame repeat.  This keeps temporal resolution high for rPPG while still
  providing a valid thermal channel for the ActivityGate spatial prior.
"""

import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

# ---------------------------------------------------------------------------
# Shared constants / helpers (mirror SIM1Loader where applicable)
# ---------------------------------------------------------------------------

CONDITIONS = ['BL', 'CD', 'ED', 'MD', 'ND', 'PD', 'RD']

DEFAULT_DRIVE_MAP = {1: 'BL', 2: 'PD', 3: 'RD', 4: 'ND',
                     5: 'CD', 6: 'ED', 7: 'MD', 8: 'FDN'}

HAAR = '/mnt/nvme/hiqbal/miniconda3/envs/py310/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
_face_cascade = None  # lazy-loaded per subprocess


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(HAAR)
    return _face_cascade


def _read_inf(inf_path):
    lines = open(inf_path).read().strip().split()
    return int(lines[0]), int(lines[1]), int(lines[2])


def _load_thermal_frames(dat_path, w, h, n_frames):
    """Load all thermal frames → (n_frames, H, W) float32 Celsius."""
    raw = np.fromfile(dat_path, dtype='<u2', count=n_frames * w * h)
    actual = len(raw) // (w * h)
    raw = raw[:actual * w * h].reshape(actual, h, w)
    return raw.astype(np.float32) * 0.01 - 273.15


def _thermal_to_uint8(frame_c):
    lo, hi = np.percentile(frame_c, [5, 99])
    norm = np.clip((frame_c - lo) / max(hi - lo, 1e-8), 0, 1)
    return (norm * 255).astype(np.uint8)


def _thermal_face_bbox(frame_c, min_area=3000, max_frame_frac=0.70):
    """Otsu + largest central blob face detection on thermal frame."""
    gray = _thermal_to_uint8(frame_c)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    h_img, w_img = frame_c.shape
    max_area = max_frame_frac * h_img * w_img
    cx_img, cy_img = w_img / 2.0, h_img / 2.0
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    best_score, best_bbox = -1.0, None
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x  = int(stats[lbl, cv2.CC_STAT_LEFT])
        y  = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        dist = ((centroids[lbl, 0] - cx_img) ** 2 +
                (centroids[lbl, 1] - cy_img) ** 2) ** 0.5
        score = area / (dist + 1.0)
        if score > best_score:
            best_score = score
            best_bbox = (x, y, bw, bh)
    return best_bbox


def _rgb_face_bbox(frame_bgr):
    """Haar cascade face detection on RGB frame. Returns (x,y,w,h) or None."""
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                     minSize=(40, 40))
    if len(faces) == 0:
        return None
    # pick largest
    areas = faces[:, 2] * faces[:, 3]
    i = np.argmax(areas)
    x, y, w, h = faces[i]
    return (int(x), int(y), int(w), int(h))


def _apply_large_box(bbox, coef, img_w, img_h):
    """Expand bbox by coef, clamp to image."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * coef, h * coef
    x1 = max(0, int(cx - nw / 2))
    y1 = max(0, int(cy - nh / 2))
    x2 = min(img_w, int(cx + nw / 2))
    y2 = min(img_h, int(cy + nh / 2))
    return x1, y1, x2 - x1, y2 - y1


def _crop_and_resize(frame, bbox, resize_w, resize_h, large_coef=1.5):
    """Crop frame using bbox (with large_box expansion) and resize."""
    H, W = frame.shape[:2]
    x, y, bw, bh = _apply_large_box(bbox, large_coef, W, H)
    crop = frame[y:y+bh, x:x+bw]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)


def _parse_drive_map_from_extracted(subj_dir):
    """Parse drive→condition from subdirectory names like '1 BL', '2 PD'."""
    mapping = {}
    pattern = re.compile(r'^(\d+)\s+([A-Z]+)$')
    if not os.path.isdir(subj_dir):
        return mapping
    for name in os.listdir(subj_dir):
        m = pattern.match(name)
        if m:
            mapping[int(m.group(1))] = m.group(2)
    return mapping


def _get_clip_labels(csv_df, drive_num, start_frame, n_frames, fps):
    """Return [hr_bpm, br_bpm, eda_raw] median over clip window, or None."""
    t_start = start_frame / fps
    t_end   = (start_frame + n_frames) / fps
    sub = csv_df[csv_df['Drive'] == drive_num].copy()
    sub = sub.assign(t_sec=sub['Time'] - 0.5)
    win = sub[(sub['t_sec'] >= t_start) & (sub['t_sec'] < t_end)]
    hr  = pd.to_numeric(win['Heart.Rate'],     errors='coerce').replace(0, np.nan).dropna()
    br  = pd.to_numeric(win['Breathing.Rate'], errors='coerce').dropna()
    eda = pd.to_numeric(win['Palm.EDA'],       errors='coerce').dropna()
    if len(hr) == 0 or len(br) == 0 or len(eda) == 0:
        return None
    return np.array([hr.median(), br.median(), eda.median()], dtype=np.float32)


def _standardize(frames):
    """Per-channel z-score normalisation over temporal axis."""
    mean = frames.mean(axis=0, keepdims=True)
    std  = frames.std(axis=0, keepdims=True)
    return (frames - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class SIM1RGBTLoader(BaseLoader):
    """Data loader for SIM1 with 4-channel RGB+Thermal input (multi-task labels)."""

    def __init__(self, name, data_path, config_data):
        self.conditions = list(config_data.PREPROCESS.SIM1.CONDITIONS)
        self.extracted_root = os.path.join(
            data_path, 'Structured Study Data', 'extracted')
        # 'fps_ratio' (proposed, Eq. 22) | 'naive' (index-based pairing ablation)
        self.alignment_mode = getattr(config_data.PREPROCESS.SIM1, 'ALIGNMENT_MODE', 'fps_ratio')
        super().__init__(name, data_path, config_data)

    # ------------------------------------------------------------------
    # get_raw_data / split_raw_data
    # ------------------------------------------------------------------

    def get_raw_data(self, data_path):
        """Enumerate valid (subject, condition) pairs that have both RGB avi1 and thermal."""
        thermal_root = os.path.join(data_path, 'Raw Thermal Data')
        entries = []
        for subj_dir in sorted(glob.glob(os.path.join(thermal_root, 'T*'))):
            subject = os.path.basename(subj_dir)
            rgb_subj_dir = os.path.join(self.extracted_root, subject)

            # Parse drive→condition from extracted folder names
            drive_map = _parse_drive_map_from_extracted(rgb_subj_dir)
            if not drive_map:
                drive_map = DEFAULT_DRIVE_MAP
            cond_to_drive = {v: k for k, v in drive_map.items()}

            for cond in self.conditions:
                inf  = os.path.join(subj_dir, f'{subject}-{cond}.inf')
                dat  = os.path.join(subj_dir, f'{subject}-{cond}.dat')
                if not (os.path.exists(inf) and os.path.exists(dat)):
                    continue

                drive_num = cond_to_drive.get(cond)
                if drive_num is None:
                    continue

                # avi1 path: extracted/TXXX/<drive_num> <COND>/TXXX-<NNN>.avi1.avi
                cond_folder = f'{drive_num} {cond}'
                avi_pattern = os.path.join(
                    rgb_subj_dir, cond_folder, f'{subject}-{drive_num:03d}.avi1.avi')
                # also try zero-padded 3 digits
                if not os.path.exists(avi_pattern):
                    # glob for any avi1 in the folder
                    matches = glob.glob(os.path.join(
                        rgb_subj_dir, cond_folder, '*.avi1.avi'))
                    avi_pattern = matches[0] if matches else None

                if avi_pattern is None or not os.path.exists(avi_pattern):
                    continue

                entries.append({
                    'index':     f'{subject}_{cond}',
                    'path':      subj_dir,
                    'subject':   subject,
                    'condition': cond,
                    'drive_num': drive_num,
                    'avi1_path': avi_pattern,
                })

        if not entries:
            raise ValueError(
                f'SIM1RGBT: no valid RGBT pairs found.\n'
                f'  Thermal root: {thermal_root}\n'
                f'  Extracted root: {self.extracted_root}\n'
                f'  Make sure you have unzipped Structured Study Data zips.')
        return entries

    def split_raw_data(self, data_dirs, begin, end):
        subjects = sorted({d['subject'] for d in data_dirs})
        n = len(subjects)
        chosen = subjects[int(begin * n):int(end * n)]
        return [d for d in data_dirs if d['subject'] in chosen]

    # ------------------------------------------------------------------
    # Core preprocessing
    # ------------------------------------------------------------------

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        entry     = data_dirs[i]
        subject   = entry['subject']
        condition = entry['condition']
        drive_num = entry['drive_num']
        avi1_path = entry['avi1_path']
        saved_filename = entry['index']

        dat_path = os.path.join(entry['path'], f'{subject}-{condition}.dat')
        inf_path = os.path.join(entry['path'], f'{subject}-{condition}.inf')
        csv_path = os.path.join(self.raw_data_path, 'R-Friendly Study Data',
                                f'{subject}.csv')

        if not os.path.exists(csv_path):
            print(f'[SKIP {saved_filename}] No CSV')
            file_list_dict[i] = []
            return

        csv_df = pd.read_csv(csv_path)

        # ---- thermal metadata ----
        try:
            n_thermal, tw, th = _read_inf(inf_path)
        except Exception as e:
            print(f'[SKIP {saved_filename}] Bad .inf: {e}')
            file_list_dict[i] = []
            return

        # ---- load all thermal frames ----
        thermal_frames = _load_thermal_frames(dat_path, tw, th, n_thermal)
        n_thermal = len(thermal_frames)

        # ---- open RGB video ----
        cap = cv2.VideoCapture(avi1_path)
        if not cap.isOpened():
            print(f'[SKIP {saved_filename}] Cannot open {avi1_path}')
            file_list_dict[i] = []
            return

        rgb_fps      = cap.get(cv2.CAP_PROP_FPS)
        n_rgb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # fps estimate for thermal from csv (same approach as SIM1Loader)
        sub_csv = csv_df[csv_df['Drive'] == drive_num]
        n_sec   = sub_csv['Time'].nunique()
        thermal_fps = n_thermal / n_sec if n_sec > 0 else 8.5

        chunk_len = config_preprocess.CHUNK_LENGTH
        resize_h  = config_preprocess.RESIZE.H
        resize_w  = config_preprocess.RESIZE.W

        # ---- face detection state ----
        rgb_bbox_last     = None
        thermal_bbox_last = None
        DETECT_FREQ       = 30   # re-detect every N RGB frames

        all_frames = []   # list of (H, W, 4) float32

        frame_idx = 0
        while True:
            ret, rgb_frame = cap.read()
            if not ret:
                break

            # thermal index alignment
            if self.alignment_mode == 'naive':
                # Ablation: naive index-based pairing (i-th RGB <-> i-th thermal
                # frame), ignoring the RGB/thermal fps ratio entirely.
                t_idx = frame_idx
            else:
                # Proposed: fps-ratio nearest-frame resampling (Eq. 22)
                t_idx = int(round(frame_idx * thermal_fps / rgb_fps))
            t_idx = max(0, min(t_idx, n_thermal - 1))
            therm = thermal_frames[t_idx]   # (th, tw) float32 Celsius

            # ---- face detection ----
            if frame_idx % DETECT_FREQ == 0:
                bbox = _rgb_face_bbox(rgb_frame)
                if bbox is not None:
                    rgb_bbox_last = bbox
                # thermal detection only updates when successful
                t_bbox = _thermal_face_bbox(therm)
                if t_bbox is not None:
                    thermal_bbox_last = t_bbox

            # ---- crop + resize RGB ----
            if rgb_bbox_last is not None:
                rgb_crop = _crop_and_resize(rgb_frame, rgb_bbox_last,
                                            resize_w, resize_h, large_coef=1.5)
            else:
                # fallback: centre crop
                y0 = int(rh * 0.1); y1 = int(rh * 0.9)
                x0 = int(rw * 0.1); x1 = int(rw * 0.9)
                rgb_crop = cv2.resize(rgb_frame[y0:y1, x0:x1],
                                      (resize_w, resize_h))
            # BGR → RGB, float32
            rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB).astype(np.float32)

            # ---- crop + resize Thermal (2D) ----
            if thermal_bbox_last is not None:
                x_t, y_t, bw_t, bh_t = _apply_large_box(
                    thermal_bbox_last, 1.5, tw, th)
                t_crop = therm[y_t:y_t+bh_t, x_t:x_t+bw_t]
                if t_crop.size == 0:
                    t_crop = therm
                therm_crop = cv2.resize(t_crop, (resize_w, resize_h),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                y0 = int(th * 0.15); y1 = int(th * 0.85)
                x0 = int(tw * 0.20); x1 = int(tw * 0.80)
                therm_crop = cv2.resize(therm[y0:y1, x0:x1],
                                        (resize_w, resize_h))
            therm_crop = np.array(therm_crop, dtype=np.float32)  # ensure 2D (H,W)

            # ---- combine RGBT: (H, W, 4) ----
            rgbt = np.concatenate([rgb_crop,
                                   therm_crop[:, :, np.newaxis]], axis=-1)
            all_frames.append(rgbt)
            frame_idx += 1

        cap.release()

        if len(all_frames) < chunk_len:
            print(f'[SKIP {saved_filename}] Too few frames: {len(all_frames)}')
            file_list_dict[i] = []
            return

        all_frames = np.array(all_frames, dtype=np.float32)   # (T, H, W, 4)

        # ---- normalise per channel ----
        # RGB channels: standardize together or per-channel
        rgb_norm   = _standardize(all_frames[:, :, :, :3])
        therm_norm = _standardize(all_frames[:, :, :, 3:4])
        all_frames = np.concatenate([rgb_norm, therm_norm], axis=-1)  # (T, H, W, 4)

        # ---- slice into clips + get labels ----
        n_clips = len(all_frames) // chunk_len
        frame_clips, label_clips = [], []

        for ci in range(n_clips):
            start_f = ci * chunk_len
            lbl = _get_clip_labels(csv_df, drive_num, start_f, chunk_len, rgb_fps)
            if lbl is None:
                continue
            frame_clips.append(all_frames[start_f:start_f + chunk_len])
            label_clips.append(lbl)

        if len(frame_clips) == 0:
            print(f'[SKIP {saved_filename}] No valid labelled clips')
            file_list_dict[i] = []
            return

        frame_clips = np.array(frame_clips)   # (n_clips, chunk_len, H, W, 4)
        label_clips = np.array(label_clips)   # (n_clips, 3)

        print(f'[{saved_filename}] {len(frame_clips)} clips, '
              f'rgb_fps={rgb_fps:.1f}, thermal_fps={thermal_fps:.2f}')

        input_name_list, _ = self._save_clips(frame_clips, label_clips, saved_filename)
        file_list_dict[i] = input_name_list

    # ------------------------------------------------------------------
    # _save_clips (same as SIM1Loader)
    # ------------------------------------------------------------------

    def _save_clips(self, frame_clips, label_clips, filename):
        os.makedirs(self.cached_path, exist_ok=True)
        input_names, label_names = [], []
        for ci in range(len(label_clips)):
            base  = os.path.join(self.cached_path, filename)
            inp_p = f'{base}_input{ci}.npy'
            lbl_p = f'{base}_label{ci}.npy'
            np.save(inp_p, frame_clips[ci])
            np.save(lbl_p, label_clips[ci])
            input_names.append(inp_p)
            label_names.append(lbl_p)
        return input_names, label_names

    @staticmethod
    def read_wave(path):
        raise NotImplementedError('SIM1RGBT uses multi-task scalar labels')
