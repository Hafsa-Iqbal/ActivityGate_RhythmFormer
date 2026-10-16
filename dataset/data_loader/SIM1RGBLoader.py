"""SIM1RGBLoader — data loader for SIM1 RGB (avi1) with multi-task labels.

Reads face RGB video from the extracted Structured Study Data, for a
configurable set of conditions (e.g. ['RD']).  Produces multi-task labels
[hr_bpm, br_bpm, eda_raw] per clip, compatible with RhythmFormerMT.

Dataset layout expected:
  <data_path>/
    Structured Study Data/
      extracted/
        T001/
          3 RD/  T001-003.avi1.avi
          ...
    R-Friendly Study Data/
      T001.csv  T002.csv  ...

Video: 640×480, ~27 fps, face is frontal and large (Haar-detectable).
Labels: 1 Hz scalar Heart.Rate, Breathing.Rate, Palm.EDA from R-Friendly Study CSV.
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
# Constants
# ---------------------------------------------------------------------------
_CASCADE_PATH = os.path.join(
    os.path.dirname(cv2.__file__), '..', '..', '..',
    'share', 'opencv4', 'haarcascades',
    'haarcascade_frontalface_default.xml')
# Fallback search if relative path doesn't resolve
if not os.path.exists(_CASCADE_PATH):
    import glob as _glob
    _candidates = _glob.glob(
        '/mnt/nvme/hiqbal/miniconda3/envs/py310/**/*haarcascade_frontalface_default.xml',
        recursive=True)
    _CASCADE_PATH = _candidates[0] if _candidates else ''

DEFAULT_DRIVE_MAP = {1: 'BL', 2: 'PD', 3: 'RD', 4: 'ND',
                     5: 'CD', 6: 'ED', 7: 'MD', 8: 'FDN'}


# ---------------------------------------------------------------------------
# Helpers (shared with SIM1Loader)
# ---------------------------------------------------------------------------

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


def _get_clip_labels(csv_df, drive_num, start_frame, n_frames, fps):
    """Return [hr_bpm, br_bpm, eda_raw] for a clip window, or None if invalid."""
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
    return np.array([hr.median(), br.median(), eda.median()], dtype=np.float32)  # shape (3,)


def _rgb_face_bbox(frame_bgr, cascade_path, last_bbox, _cascade_cache={}):
    """Detect face in RGB frame using Haar cascade; fall back to last_bbox.

    The cascade is loaded once per process and cached in _cascade_cache to
    avoid repeated disk reads without storing it as an instance attribute
    (which would break multiprocessing pickling).
    """
    if cascade_path not in _cascade_cache:
        _cascade_cache[cascade_path] = cv2.CascadeClassifier(cascade_path)
    cascade = _cascade_cache[cascade_path]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(50, 50))
    if len(faces) == 0:
        return last_bbox, False
    # Pick largest face
    best = max(faces, key=lambda f: f[2] * f[3])
    return tuple(int(v) for v in best), True


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class SIM1RGBLoader(BaseLoader):
    """Data loader for SIM1 RGB (avi1) face video — multi-task HR/BR/EDA."""

    def __init__(self, name, data_path, config_data):
        self.conditions = list(config_data.PREPROCESS.SIM1.CONDITIONS)
        # Validate cascade path at init time, but don't store the object
        # (cv2.CascadeClassifier is not picklable — instantiate per subprocess)
        if not os.path.exists(_CASCADE_PATH):
            raise RuntimeError(f'Haar cascade not found at {_CASCADE_PATH}')
        super().__init__(name, data_path, config_data)

    # ------------------------------------------------------------------
    # get_raw_data / split_raw_data
    # ------------------------------------------------------------------

    def get_raw_data(self, data_path):
        """Enumerate all valid (subject, condition) pairs from extracted dir."""
        extracted_root = os.path.join(
            data_path, 'Structured Study Data', 'extracted')
        entries = []
        for subj_dir in sorted(glob.glob(os.path.join(extracted_root, 'T*'))):
            subject = os.path.basename(subj_dir)
            for cond in self.conditions:
                # Folder name: "N COND"  e.g. "3 RD"
                cond_dirs = glob.glob(os.path.join(subj_dir, f'* {cond}'))
                if not cond_dirs:
                    continue
                cond_dir = cond_dirs[0]
                avi_files = glob.glob(os.path.join(cond_dir, '*.avi1.avi'))
                if not avi_files:
                    continue
                entries.append({
                    'index':     f'{subject}_{cond}',
                    'path':      cond_dir,
                    'avi_path':  avi_files[0],
                    'subject':   subject,
                    'condition': cond,
                })
        if not entries:
            raise ValueError(
                f'SIM1RGB: no (subject, condition) pairs found under {extracted_root}')
        return entries

    def split_raw_data(self, data_dirs, begin, end):
        """Split by subject so all conditions of a subject stay together."""
        subjects = sorted({d['subject'] for d in data_dirs})
        n = len(subjects)
        chosen = subjects[int(begin * n):int(end * n)]
        return [d for d in data_dirs if d['subject'] in chosen]

    # ------------------------------------------------------------------
    # Core preprocessing
    # ------------------------------------------------------------------

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,
                                      file_list_dict):
        """Process one (subject, condition) pair → save clips + labels."""
        entry     = data_dirs[i]
        subject   = entry['subject']
        condition = entry['condition']
        saved_filename = entry['index']   # e.g. "T001_RD"
        avi_path  = entry['avi_path']

        # ---- drive number for CSV lookup ----
        zip_path = os.path.join(self.raw_data_path, 'Structured Study Data',
                                f'{subject}.zip')
        drive_map  = _parse_drive_map_from_zip(zip_path) or dict(DEFAULT_DRIVE_MAP)
        cond_to_drive = {v: k for k, v in drive_map.items()}
        drive_num  = cond_to_drive.get(condition)
        if drive_num is None:
            print(f'[SKIP {saved_filename}] No drive mapping for {condition}')
            file_list_dict[i] = []
            return

        # ---- labels CSV ----
        csv_path = os.path.join(self.raw_data_path, 'R-Friendly Study Data',
                                f'{subject}.csv')
        if not os.path.exists(csv_path):
            print(f'[SKIP {saved_filename}] No CSV found')
            file_list_dict[i] = []
            return
        csv_df = pd.read_csv(csv_path)

        # ---- open video ----
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            print(f'[SKIP {saved_filename}] Cannot open {avi_path}')
            file_list_dict[i] = []
            return
        fps        = cap.get(cv2.CAP_PROP_FPS) or 27.0
        n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resize_h   = config_preprocess.RESIZE.H
        resize_w   = config_preprocess.RESIZE.W
        chunk_len  = config_preprocess.CHUNK_LENGTH
        det_freq   = config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY

        # ---- read all frames with face crop ----
        all_frames = []   # list of (H, W, 3) uint8
        last_bbox  = None
        n_detected = n_held = n_fallback = 0

        for fi in range(n_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if fi % det_freq == 0 or last_bbox is None:
                bbox, found = _rgb_face_bbox(frame_bgr, _CASCADE_PATH, last_bbox)
                if found:
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

            if last_bbox is not None:
                x, y, bw, bh = last_bbox
                # 15% margin
                mx, my = int(bw * 0.15), int(bh * 0.15)
                h_img, w_img = frame_bgr.shape[:2]
                x1 = max(0, x - mx);  y1 = max(0, y - my)
                x2 = min(w_img, x + bw + mx);  y2 = min(h_img, y + bh + my)
                crop = frame_bgr[y1:y2, x1:x2]
            else:
                # Centre-crop fallback
                h_img, w_img = frame_bgr.shape[:2]
                crop = frame_bgr[int(h_img*0.1):int(h_img*0.9),
                                 int(w_img*0.2):int(w_img*0.8)]

            if crop.size == 0:
                crop = frame_bgr

            resized = cv2.resize(crop, (resize_w, resize_h),
                                 interpolation=cv2.INTER_LINEAR)
            all_frames.append(resized)   # (H, W, 3) uint8 BGR

        cap.release()

        total = n_detected + n_held + n_fallback
        coverage = 100.0 * (n_detected + n_held) / max(total, 1)
        print(f'[{saved_filename}] frames={len(all_frames)} '
              f'face det: detected={n_detected} held={n_held} '
              f'fallback={n_fallback}  coverage={coverage:.1f}%')

        if len(all_frames) < chunk_len:
            print(f'[SKIP {saved_filename}] Too few frames: {len(all_frames)}')
            file_list_dict[i] = []
            return

        all_frames = np.array(all_frames, dtype=np.float32)   # (T, H, W, 3)

        # ---- normalise ----
        processed_frames = self._apply_data_type(all_frames,
                                                  config_preprocess.DATA_TYPE)

        # ---- build clips and labels ----
        n_clips     = len(all_frames) // chunk_len
        frame_clips = []
        label_clips = []

        for ci in range(n_clips):
            start_f = ci * chunk_len
            lbl = _get_clip_labels(csv_df, drive_num, start_f, chunk_len, fps)
            if lbl is None:
                continue
            frame_clips.append(processed_frames[start_f:start_f + chunk_len])
            label_clips.append(lbl)

        if not frame_clips:
            print(f'[SKIP {saved_filename}] No valid clips with labels')
            file_list_dict[i] = []
            return

        frame_clips = np.array(frame_clips)   # (n_clips, chunk_len, H, W, 3)
        label_clips = np.array(label_clips)   # (n_clips, 3)

        input_name_list, _ = self._save_clips(frame_clips, label_clips,
                                              saved_filename)
        file_list_dict[i] = input_name_list

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_data_type(frames, data_types):
        """Apply standardization / diff-norm (same convention as SIM1Loader)."""
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
        """Save (n_clips, chunk_len, H, W, C) input and (n_clips, 3) labels."""
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
        raise NotImplementedError('SIM1RGB uses scalar HR labels, not waveforms')
