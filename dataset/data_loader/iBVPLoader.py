"""The dataloader for iBVP dataset — unified for RGB, RGBT, and Thermal-only modes.

Controlled by config field: PREPROCESS.IBVP.DATA_MODE ('RGB', 'RGBT', 'T', or 'T_fake_rgb').
"""

import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd


class iBVPLoader(BaseLoader):
    """The unified data loader for the iBVP dataset (RGB / RGBT / Thermal)."""

    def __init__(self, name, data_path, config_data):
        """Initializes an iBVP dataloader.
        Args:
            data_path(str): path of a folder which stores raw video and bvp data.
            e.g. data_path should be "RawData" for below dataset structure:
            -----------------
                 RawData/
                 |   |-- p01_a/
                 |       |-- rgb/*.bmp (timestamped BMPs)
                 |       |-- thermal/*.raw (thermal frames)
                 |       |-- p01_a_bvp.csv
                 |   |-- p02_b/
                 |       |-- rgb/*.bmp
                 |       |-- thermal/*.raw
                 |       |-- p02_b_bvp.csv
                 |...
            -----------------
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.data_mode = config_data.PREPROCESS.IBVP.DATA_MODE  # 'RGB', 'RGBT', or 'T'
        # 'timestamp' (proposed nearest-timestamp alignment) | 'naive' (index-based pairing ablation)
        self.alignment_mode = getattr(config_data.PREPROCESS.IBVP, 'ALIGNMENT_MODE', 'timestamp')
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (For iBVP dataset)."""
        data_dirs = sorted(glob.glob(data_path + os.sep + "p*_[a-d]"))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": os.path.basename(data_dir), "path": data_dir}
                for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = [data_dirs[i] for i in choose_range]
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]['index']
        data_mode = self.data_mode

        try:
            rgb_folder = os.path.join(data_dirs[i]['path'], "rgb")
            thermal_folder = os.path.join(data_dirs[i]['path'], "thermal")

            if data_mode == 'RGB':
                frames = self.read_video(rgb_folder)
            elif data_mode == 'T':
                frames = self.read_thermal_video(thermal_folder)
            elif data_mode == 'RGBT':
                rgb_frames, rgb_ts = self.read_video_with_timestamps(rgb_folder)
                thermal_frames, thm_ts = self.read_thermal_video_with_timestamps(thermal_folder)
                if len(rgb_frames) == 0 or len(thermal_frames) == 0:
                    file_list_dict[i] = []
                    return
                if self.alignment_mode == 'naive':
                    # Ablation: naive index-based pairing (i-th RGB <-> i-th thermal frame),
                    # ignoring acquisition timestamps entirely.
                    rgb_frames, thermal_frames, _ = self._align_modalities(rgb_frames, thermal_frames)
                else:
                    rgb_frames, thermal_frames = self._align_modalities_by_timestamp(
                        rgb_frames, rgb_ts, thermal_frames, thm_ts)
                # Crop thermal to match RGB spatial height before concat
                thermal_frames = thermal_frames[:, :rgb_frames.shape[1], :, :]
                frames = np.concatenate([rgb_frames, thermal_frames], axis=-1)
            elif data_mode == 'T_fake_rgb':
                frames = self.read_thermal_as_fakergb(thermal_folder)
            else:
                raise ValueError(f"Unsupported DATA_MODE: {data_mode}")
        except Exception as e:
            print(f"[ERROR {i} {saved_filename}] Frame loading failed: {e}")
            file_list_dict[i] = []
            return

        # Read Labels (iBVP CSV)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvp_file = os.path.join(data_dirs[i]['path'], f"{saved_filename}_bvp.csv")
            try:
                bvps = self.read_wave(bvp_file)
            except Exception as e:
                print(f"[SKIP {i} {saved_filename}] Error reading BVP: {e}")
                file_list_dict[i] = []
                return

        # Resample BVP to match frame count
        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        try:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        except Exception as e:
            print(f"[SKIP {i} {saved_filename}] Preprocess error: {e}")
            file_list_dict[i] = []
            return

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def _align_modalities(rgb_frames, thermal_frames, chunk_length=None):
        """Align RGB and thermal frame arrays to the same temporal length."""
        rgb_length = rgb_frames.shape[0]
        thermal_length = thermal_frames.shape[0]
        target_length = min(rgb_length, thermal_length)

        if chunk_length is not None and target_length < chunk_length:
            if rgb_length >= chunk_length and thermal_length > 0 and thermal_length < rgb_length:
                repeat = int(np.ceil(rgb_length / thermal_length))
                thermal_frames = np.repeat(thermal_frames, repeat, axis=0)[:rgb_length]
            elif thermal_length >= chunk_length and rgb_length > 0 and rgb_length < thermal_length:
                repeat = int(np.ceil(thermal_length / rgb_length))
                rgb_frames = np.repeat(rgb_frames, repeat, axis=0)[:thermal_length]
            target_length = min(rgb_frames.shape[0], thermal_frames.shape[0])

        rgb_frames = rgb_frames[:target_length]
        thermal_frames = thermal_frames[:target_length]
        return rgb_frames, thermal_frames, target_length

    @staticmethod
    def _align_modalities_by_timestamp(rgb_frames, rgb_ts, thermal_frames, thm_ts):
        """Align thermal to RGB using nearest-timestamp matching.

        For each RGB frame, picks the thermal frame with the closest timestamp.
        Returns: (rgb_frames unchanged, aligned_thermal_frames), both length T=len(rgb_frames).
        """
        # searchsorted gives insertion point; check both neighbours for true nearest
        idx = np.searchsorted(thm_ts, rgb_ts)
        idx = np.clip(idx, 0, len(thm_ts) - 1)
        prev_idx = np.maximum(idx - 1, 0)
        dist_next = np.abs(thm_ts[idx] - rgb_ts)
        dist_prev = np.abs(thm_ts[prev_idx] - rgb_ts)
        nearest = np.where(dist_prev < dist_next, prev_idx, idx)
        return rgb_frames, thermal_frames[nearest]

    @staticmethod
    def read_video_with_timestamps(video_folder):
        """Reads BMP image sequence, returns (frames(T,H,W,3), timestamps(T,) in ms)."""
        bmp_files = glob.glob(os.path.join(video_folder, '*.bmp'))
        bmp_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        frames, timestamps = [], []
        for bmp_file in bmp_files:
            frame = cv2.imread(bmp_file)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                timestamps.append(int(os.path.basename(bmp_file).split('.')[0]))
        return np.asarray(frames), np.asarray(timestamps, dtype=np.int64)

    @staticmethod
    def read_thermal_video_with_timestamps(thermal_folder):
        """Reads raw thermal image sequence, returns (frames(T,H,W,1), timestamps(T,) in ms)."""
        im_width, im_height = 640, 512
        raw_files = glob.glob(os.path.join(thermal_folder, '*.raw'))
        raw_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        frames, timestamps = [], []
        for raw_path in raw_files:
            data = np.fromfile(raw_path, dtype=np.uint16, count=im_width * im_height)
            if len(data) == im_width * im_height:
                frame = ((data.reshape(im_height, im_width).astype(np.float32) * 0.04) - 273.15)
                frames.append(frame)
                timestamps.append(int(os.path.splitext(os.path.basename(raw_path))[0]))
        return np.expand_dims(np.array(frames), -1), np.asarray(timestamps, dtype=np.int64)

    @staticmethod
    def read_video(video_folder):
        """Reads BMP image sequence (iBVP-style), returns frames(T, H, W, 3)."""
        bmp_files = glob.glob(os.path.join(video_folder, '*.bmp'))
        bmp_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # Timestamp sort

        frames = []
        for bmp_file in bmp_files:
            frame = cv2.imread(bmp_file)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        return np.asarray(frames)

    @staticmethod
    def read_thermal_video(thermal_folder):
        """Reads raw thermal image sequence (iBVP-style), returns frames(T, H, W, 1)."""
        im_width, im_height = 640, 512
        raw_files = sorted(glob.glob(os.path.join(thermal_folder, '*.raw')))
        frames = []
        for raw_path in raw_files:
            data = np.fromfile(raw_path, dtype=np.uint16, count=im_width * im_height)
            if len(data) == im_width * im_height:
                frame = ((data.reshape(im_height, im_width).astype(np.float32) * 0.04) - 273.15)
                frames.append(frame)
        return np.expand_dims(np.array(frames), -1)

    @staticmethod
    def read_thermal_as_fakergb(thermal_folder):
        """Reads raw thermal sequence, converts to Celsius, normalizes per-clip to [0,255],
        and stacks as 3-channel fake RGB for use with standard RGB model pipelines.

        Conversion: raw_uint16 * 0.04 - 273.15 = Celsius
        Normalization: per-clip min-max → [0, 255] (preserves inter-frame dynamics)
        Output: (T, H, W, 3) float32, all 3 channels identical (thermal grayscale as fake RGB)
        """
        im_width, im_height = 640, 512
        raw_files = sorted(glob.glob(os.path.join(thermal_folder, '*.raw')))
        frames = []
        for raw_path in raw_files:
            data = np.fromfile(raw_path, dtype=np.uint16, count=im_width * im_height)
            if len(data) == im_width * im_height:
                frame_c = (data.reshape(im_height, im_width).astype(np.float32) * 0.04) - 273.15
                frames.append(frame_c)
        if not frames:
            return np.zeros((0, im_height, im_width, 3), dtype=np.float32)
        frames_arr = np.array(frames)  # (T, H, W)
        # Per-clip normalization to preserve inter-frame temperature differences
        t_min = frames_arr.min()
        t_max = frames_arr.max()
        frames_norm = (frames_arr - t_min) / (t_max - t_min + 1e-7) * 255.0
        # Stack as 3-channel fake RGB: (T, H, W) -> (T, H, W, 3)
        return np.stack([frames_norm, frames_norm, frames_norm], axis=-1)

    @staticmethod
    def read_wave(bvp_file):
        """Reads BVP CSV signal file (iBVP-style, simple like UBFC)."""
        df = pd.read_csv(bvp_file)
        if 'BVP' in df.columns:
            return df['BVP'].values
        elif len(df.columns) > 0:
            return df.iloc[:, 0].values
        raise ValueError(f"No BVP data found in {bvp_file}")
