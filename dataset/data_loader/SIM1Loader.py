"""SIM1Loader — data loader for the SIMULATOR STUDY I (SIM1) dataset.

Produces multi-task labels: [hr_bpm, br_bpm, eda_raw] per clip.

Dataset layout expected:
  <data_path>/
    Raw Thermal Data/
      T001/  T001-BL.dat  T001-BL.inf  T001-PD.dat  ...
    R-Friendly Study Data/
      T001.csv  T002.csv  ...
    Structured Study Data/
      T001.zip  T002.zip  ...   (used only to derive drive→condition mapping)

Thermal encoding:  T_celsius = uint16 * 0.01 - 273.15
Thermal FPS:       ~8.5 Hz  (n_frames / n_seconds_from_CSV)
"""

import glob
import os
import re
import zipfile

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Condition catalogue
# ---------------------------------------------------------------------------
CONDITIONS = ['BL', 'CD', 'ED', 'MD', 'ND', 'PD', 'RD']

# Default drive→condition map (overridden per-subject from zip if available)
DEFAULT_DRIVE_MAP = {1: 'BL', 2: 'PD', 3: 'RD', 4: 'ND',
                     5: 'CD', 6: 'ED', 7: 'MD', 8: 'FDN'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_inf(inf_path):
    """Parse .inf header → (n_frames, width, height)."""
    lines = open(inf_path).read().strip().split()
    return int(lines[0]), int(lines[1]), int(lines[2])


def _load_frames_chunk(dat_path, w, h, start, count):
    """Read `count` uint16 frames starting at `start` from raw .dat file.

    Returns float32 ndarray of shape (count, h, w) in Celsius.
    """
    frame_pixels = w * h
    offset_bytes = start * frame_pixels * 2          # uint16 = 2 bytes
    raw = np.fromfile(dat_path, dtype='<u2',
                      count=count * frame_pixels,
                      offset=offset_bytes)
    n_read = min(len(raw) // frame_pixels, count)
    raw = raw[:n_read * frame_pixels].reshape(n_read, h, w)
    return raw.astype(np.float32) * 0.01 - 273.15   # Kelvin*100 → Celsius


def _thermal_to_uint8(frame_c):
    """Percentile-normalise a Celsius frame to 0-255 uint8 (single channel)."""
    lo, hi = np.percentile(frame_c, [5, 99])
    norm = np.clip((frame_c - lo) / max(hi - lo, 1e-8), 0, 1)
    return (norm * 255).astype(np.uint8)


def _thermal_to_uint8_bgr(frame_c):
    """Percentile-normalise a Celsius frame to 0-255 uint8 BGR (kept for debug scripts)."""
    return cv2.cvtColor(_thermal_to_uint8(frame_c), cv2.COLOR_GRAY2BGR)


def _thermal_face_bbox(frame_c, min_area=3000, max_frame_frac=0.70):
    """Detect face ROI in a thermal frame using Otsu threshold + largest central blob.

    SIM1 driving simulator has extreme cold (LCD screens, -35°C) and hot
    (lamps, 65°C) regions.  Haar cascades trained on visible-light fail here.
    Instead: percentile-normalise → Otsu threshold → find warm blobs →
    pick the largest blob nearest the image centre (face is roughly centred).

    Args:
        frame_c        : (H, W) float32 Celsius array.
        min_area       : minimum blob area in pixels (rejects noise).
        max_frame_frac : blobs covering > this fraction of frame area are
                         discarded (rejects frames where hot lamps merge with
                         the face into one giant blob).

    Returns:
        (x, y, w, h) bounding box  or  None if no suitable blob found.
    """
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


def _parse_drive_map_from_zip(zip_path):
    """Extract drive→condition mapping from zip folder names 'N COND/'."""
    mapping = {}
    pattern = re.compile(r'^[^/]+/(\d+)\s+([A-Z]+)/$')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                m = pattern.match(name)
                if m:
                    mapping[int(m.group(1))] = m.group(2)
    except Exception:
        pass
    return mapping


def _get_fps(csv_df, drive_num, n_frames):
    """Estimate FPS = n_frames / n_seconds for a given drive."""
    sub = csv_df[csv_df['Drive'] == drive_num]
    n_sec = sub['Time'].nunique()
    return n_frames / n_sec if n_sec > 0 else 8.5


def _get_clip_labels(csv_df, drive_num, start_frame, n_frames, fps):
    """Return [hr_bpm, br_bpm, eda_raw] for a clip window, or None if invalid."""
    t_start = start_frame / fps
    t_end   = (start_frame + n_frames) / fps

    sub = csv_df[csv_df['Drive'] == drive_num].copy()
    sub = sub.assign(t_sec=sub['Time'] - 0.5)   # 1-based Time → 0-based seconds
    win = sub[(sub['t_sec'] >= t_start) & (sub['t_sec'] < t_end)]

    hr  = pd.to_numeric(win['Heart.Rate'],     errors='coerce').replace(0, np.nan).dropna()
    br  = pd.to_numeric(win['Breathing.Rate'], errors='coerce').dropna()
    eda = pd.to_numeric(win['Palm.EDA'],       errors='coerce').dropna()

    if len(hr) == 0 or len(br) == 0 or len(eda) == 0:
        return None
    return np.array([hr.median(), br.median(), eda.median()], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class SIM1Loader(BaseLoader):
    """Data loader for SIM1 thermal multi-task dataset (HR, BR, EDA)."""

    def __init__(self, name, data_path, config_data):
        self.data_mode  = config_data.PREPROCESS.SIM1.DATA_MODE   # 'T'
        self.conditions = list(config_data.PREPROCESS.SIM1.CONDITIONS)
        super().__init__(name, data_path, config_data)

    # ------------------------------------------------------------------
    # get_raw_data / split_raw_data
    # ------------------------------------------------------------------

    def get_raw_data(self, data_path):
        """Enumerate all valid (subject, condition) pairs."""
        thermal_root = os.path.join(data_path, 'Raw Thermal Data')
        entries = []
        for subj_dir in sorted(glob.glob(os.path.join(thermal_root, 'T*'))):
            subject = os.path.basename(subj_dir)
            for cond in self.conditions:
                inf = os.path.join(subj_dir, f'{subject}-{cond}.inf')
                dat = os.path.join(subj_dir, f'{subject}-{cond}.dat')
                if os.path.exists(inf) and os.path.exists(dat):
                    entries.append({
                        'index':     f'{subject}_{cond}',
                        'path':      subj_dir,
                        'subject':   subject,
                        'condition': cond,
                    })
        if not entries:
            raise ValueError(f'SIM1: no (subject, condition) pairs found under {thermal_root}')
        return entries

    def split_raw_data(self, data_dirs, begin, end):
        """Split by unique subjects so all conditions of a subject stay together."""
        subjects = sorted({d['subject'] for d in data_dirs})
        n = len(subjects)
        chosen = subjects[int(begin * n):int(end * n)]
        return [d for d in data_dirs if d['subject'] in chosen]

    # ------------------------------------------------------------------
    # Core preprocessing
    # ------------------------------------------------------------------

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Process one (subject, condition) pair → save clips + labels."""
        entry     = data_dirs[i]
        subject   = entry['subject']
        condition = entry['condition']
        saved_filename = entry['index']   # e.g. "T001_BL"

        thermal_dir   = entry['path']
        inf_path = os.path.join(thermal_dir, f'{subject}-{condition}.inf')
        dat_path = os.path.join(thermal_dir, f'{subject}-{condition}.dat')

        csv_path = os.path.join(self.raw_data_path, 'R-Friendly Study Data',
                                f'{subject}.csv')
        zip_path = os.path.join(self.raw_data_path, 'Structured Study Data',
                                f'{subject}.zip')

        # ---- labels CSV ----
        if not os.path.exists(csv_path):
            print(f'[SKIP {saved_filename}] No CSV found')
            file_list_dict[i] = []
            return
        csv_df = pd.read_csv(csv_path)

        # ---- drive→condition mapping ----
        drive_map  = _parse_drive_map_from_zip(zip_path) if os.path.exists(zip_path) else {}
        if not drive_map:
            drive_map = DEFAULT_DRIVE_MAP
        cond_to_drive = {v: k for k, v in drive_map.items()}
        drive_num = cond_to_drive.get(condition)
        if drive_num is None:
            print(f'[SKIP {saved_filename}] No drive mapping for condition {condition}')
            file_list_dict[i] = []
            return

        # ---- thermal metadata ----
        try:
            n_frames, w, h = _read_inf(inf_path)
        except Exception as e:
            print(f'[SKIP {saved_filename}] Bad .inf: {e}')
            file_list_dict[i] = []
            return

        fps = _get_fps(csv_df, drive_num, n_frames)
        chunk_len = config_preprocess.CHUNK_LENGTH
        resize_h  = config_preprocess.RESIZE.H
        resize_w  = config_preprocess.RESIZE.W

        # ---- face detection state ----
        last_bbox = None        # (x, y, bw, bh) last successful detection
        n_detected = 0
        n_held     = 0
        n_fallback = 0

        # ---- accumulate frames ----
        all_frames = []    # list of (resize_h, resize_w, 1) float32 clips

        READ_CHUNK = 64    # stream in 64-frame I/O chunks to limit RAM
        frame_idx  = 0

        while frame_idx < n_frames:
            count = min(READ_CHUNK, n_frames - frame_idx)
            chunk = _load_frames_chunk(dat_path, w, h, frame_idx, count)

            for local_i in range(chunk.shape[0]):
                frame_c = chunk[local_i]   # (H, W) float32 Celsius

                # Re-run detection every 30 frames (or until first success)
                if frame_idx % 30 == 0 or last_bbox is None:
                    bbox = _thermal_face_bbox(frame_c)
                    if bbox is not None:
                        last_bbox = bbox
                        n_detected += 1
                    elif last_bbox is not None:
                        n_held += 1
                    else:
                        n_fallback += 1
                else:
                    if last_bbox is not None:
                        n_held += 1
                    else:
                        n_fallback += 1

                # Apply face crop or fallback
                if last_bbox is not None:
                    x0, y0, bw, bh = last_bbox
                    mx, my = int(bw * 0.15), int(bh * 0.15)
                    x1 = max(0, x0 - mx);  y1 = max(0, y0 - my)
                    x2 = min(w, x0 + bw + mx);  y2 = min(h, y0 + bh + my)
                    crop = frame_c[y1:y2, x1:x2]
                else:
                    # Fallback: centre 60% × 70% of frame
                    cx, cy = w // 2, h // 2
                    crop = frame_c[int(h*0.15):int(h*0.85), int(w*0.20):int(w*0.80)]

                if crop.size == 0:
                    crop = frame_c[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

                resized = cv2.resize(crop, (resize_w, resize_h),
                                     interpolation=cv2.INTER_LINEAR)
                all_frames.append(resized[:, :, np.newaxis])   # (H, W, 1)
                frame_idx += 1

        total = n_detected + n_held + n_fallback
        coverage = 100.0 * (n_detected + n_held) / max(total, 1)
        print(f'[{saved_filename}] face det: detected={n_detected} '
              f'held={n_held} fallback={n_fallback}  coverage={coverage:.1f}%')

        if len(all_frames) < chunk_len:
            print(f'[SKIP {saved_filename}] Too few frames: {len(all_frames)}')
            file_list_dict[i] = []
            return

        all_frames = np.array(all_frames, dtype=np.float32)   # (T, H, W, 1)

        # ---- normalise frames (Standardized) ----
        data_types = config_preprocess.DATA_TYPE
        processed_frames = self._apply_data_type(all_frames, data_types)

        # ---- build clips and labels ----
        n_clips = len(all_frames) // chunk_len
        frame_clips = []
        label_clips = []

        for ci in range(n_clips):
            start_f = ci * chunk_len
            lbl = _get_clip_labels(csv_df, drive_num, start_f, chunk_len, fps)
            if lbl is None:
                continue
            frame_clips.append(processed_frames[start_f:start_f + chunk_len])
            label_clips.append(lbl)

        if len(frame_clips) == 0:
            print(f'[SKIP {saved_filename}] No valid clips with labels')
            file_list_dict[i] = []
            return

        frame_clips = np.array(frame_clips)   # (n_clips, chunk_len, H, W, 1)
        label_clips = np.array(label_clips)   # (n_clips, 3)

        input_name_list, _ = self._save_clips(frame_clips, label_clips, saved_filename)
        file_list_dict[i] = input_name_list

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_data_type(frames, data_types):
        """Apply standardization over the temporal axis (same as BaseLoader)."""
        results = []
        for dt in data_types:
            if dt == 'Raw':
                results.append(frames.copy())
            elif dt == 'Standardized':
                mean = frames.mean(axis=0, keepdims=True)
                std  = frames.std(axis=0, keepdims=True)
                results.append((frames - mean) / (std + 1e-8))
            elif dt == 'DiffNormalized':
                diff = np.diff(frames, axis=0)
                diff = np.concatenate([diff[:1], diff], axis=0)
                d_std = diff.std(axis=0, keepdims=True)
                results.append(diff / (d_std + 1e-8))
            else:
                results.append(frames.copy())
        return np.concatenate(results, axis=-1)   # (T, H, W, C_total)

    def _save_clips(self, frame_clips, label_clips, filename):
        """Save (n_clips, chunk_len, H, W, C) input and (n_clips, 3) label arrays."""
        os.makedirs(self.cached_path, exist_ok=True)
        input_names, label_names = [], []
        for ci in range(len(label_clips)):
            base  = os.path.join(self.cached_path, f'{filename}')
            inp_p = f'{base}_input{ci}.npy'
            lbl_p = f'{base}_label{ci}.npy'
            np.save(inp_p, frame_clips[ci])    # (chunk_len, H, W, C)
            np.save(lbl_p, label_clips[ci])    # (3,)
            input_names.append(inp_p)
            label_names.append(lbl_p)
        return input_names, label_names

    # ------------------------------------------------------------------
    # read_wave stub (required by BaseLoader interface, not used here)
    # ------------------------------------------------------------------

    @staticmethod
    def read_wave(path):
        raise NotImplementedError('SIM1 uses multi-task scalar labels, not waveforms')
