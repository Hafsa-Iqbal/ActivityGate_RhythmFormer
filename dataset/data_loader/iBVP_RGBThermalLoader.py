import glob
import os
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

class iBVP_RGBThermalLoader(BaseLoader):
    def __init__(self, name, data_path, config_data):
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
        import traceback
        try:
            filename = os.path.split(data_dirs[i]['path'])[-1]
            saved_filename = data_dirs[i]['index']
            #print(f"[START {i:2d} {saved_filename}] Beginning preprocessing...")

            # Read Frames (iBVP BMP sequence)
            rgb_folder = os.path.join(data_dirs[i]['path'], "rgb")
            # quick diagnostic: count files before reading
            rgb_files = sorted(glob.glob(os.path.join(rgb_folder, '*.bmp')))
            #print(f"[LOAD {i:2d} {saved_filename}] Loading {len(rgb_files)} RGB frames...")
            rgb_frames = self.read_video(rgb_folder)
            #print(f"[LOAD {i:2d} {saved_filename}] RGB loaded: {rgb_frames.shape}")

            thermal_folder = os.path.join(data_dirs[i]['path'], "thermal")
            thermal_files = sorted(glob.glob(os.path.join(thermal_folder, '*.raw')))
            #print(f"[LOAD {i:2d} {saved_filename}] Loading {len(thermal_files)} thermal frames...")
            thermal_frames = self.read_thermal_video(thermal_folder)
            #print(f"[LOAD {i:2d} {saved_filename}] Thermal loaded: {thermal_frames.shape}")
        except Exception as e:
            #print(f"[CRASH {i:2d}] Failed during frame loading: {e}")
            #print(f"[CRASH {i:2d}] Traceback: {traceback.format_exc()}")
            file_list_dict[i] = []
            return

        # BVP path for diagnostic
        bvp_file = os.path.join(data_dirs[i]['path'], f"{saved_filename}_bvp.csv")
        #print(f"[COUNT {i} {saved_filename}] RGB_files={len(rgb_files)}, Thermal_files={len(thermal_files)}, BVP_exists={os.path.exists(bvp_file)}")
        
            
        rgb_length, rgb_height, rgb_width, rgb_ch = rgb_frames.shape
        thermal_length, t_height, t_width, t_ch = thermal_frames.shape
        target_length = min(rgb_length, thermal_length)
        # If one modality is shorter than chunk length this can lead to 0 clips for fused data.
        chunk_len = config_preprocess.CHUNK_LENGTH if hasattr(config_preprocess, 'CHUNK_LENGTH') else None
        if chunk_len is not None and target_length < chunk_len:
            # Try to upsample the shorter modality by repeating frames so we can form at least one chunk
            if rgb_length >= chunk_len and thermal_length > 0 and thermal_length < rgb_length:
                repeat = int(np.ceil(rgb_length / thermal_length))
                thermal_frames = np.repeat(thermal_frames, repeat, axis=0)[:rgb_length, :rgb_height, :, :]
                target_length = min(rgb_frames.shape[0], thermal_frames.shape[0])
                #print(f"[UPSAMPLE] Thermal frames repeated x{repeat} to match RGB length: new_target={target_length}")
            elif thermal_length >= chunk_len and rgb_length > 0 and rgb_length < thermal_length:
                repeat = int(np.ceil(thermal_length / rgb_length))
                rgb_frames = np.repeat(rgb_frames, repeat, axis=0)[:thermal_length, ...]
                target_length = min(rgb_frames.shape[0], thermal_frames.shape[0])
                #print(f"[UPSAMPLE] RGB frames repeated x{repeat} to match Thermal length: new_target={target_length}")

        # If after possible upsampling there are still no frames, skip this sample
        if target_length == 0:
            #print(f"[SKIP {i} {saved_filename}] No frames after alignment")
            file_list_dict[i] = []
            return

        rgb_frames = rgb_frames[:target_length, ...]
        thermal_frames = thermal_frames[:target_length, :rgb_height, :, :] 
        concat_frames = np.concatenate([rgb_frames, thermal_frames], axis=-1)
        #print(f"concatenate frames shapes {frames.shape}]")

        # Read Labels (iBVP CSV)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(concat_frames, fs=self.config_data.FS)
        else:
            bvp_file = os.path.join(data_dirs[i]['path'], f"{saved_filename}_bvp.csv")
            try:
                bvps = self.read_wave(bvp_file)
            except Exception as e:
                #print(f"[SKIP {i} {saved_filename}] Error reading BVP: {e}")
                file_list_dict[i] = []
                return

        # Validate BVP signal
        if bvps is None or len(bvps) == 0:
            #print(f"[SKIP {i} {saved_filename}] No BVP - len={0 if bvps is None else len(bvps)}")
            file_list_dict[i] = []
            return

        # Resample BVP to match frame length (prevent zero clips when bvps shorter than chunk length)
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        chunk_len = config_preprocess.CHUNK_LENGTH if hasattr(config_preprocess, 'CHUNK_LENGTH') else None
        #print(f"[DEBUG {i} {saved_filename}] frames={target_length}, bvp_len={len(bvps)}, chunk_len={chunk_len}")

        try:
            frames_clips, bvps_clips = self.preprocess(concat_frames, bvps, config_preprocess)
        except Exception as e:
            import traceback
            #print(f"[SKIP {i} {saved_filename}] Preprocess error: {e}")
            #print(f"[ERROR DETAILS] {traceback.format_exc()}")
            file_list_dict[i] = []
            return

        # Report number of generated clips for this sample
        num_clips = 0 if bvps_clips is None or len(bvps_clips) == 0 else len(bvps_clips)
        print(f"[PROCESS {i} {saved_filename}] Generated clips={num_clips}")

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

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
        im_width, im_height = 640, 512
        raw_files = sorted(glob.glob(os.path.join(thermal_folder, '*.raw')))
        frames = []
        for raw_path in raw_files:
            data = np.fromfile(raw_path, dtype=np.uint16, count=im_width*im_height)
            if len(data) == im_width*im_height:
                frame = ((data.reshape(im_height, im_width).astype(np.float32) * 0.04) - 273.15)
                frames.append(frame)
        return np.expand_dims(np.array(frames), -1)


    @staticmethod
    def read_wave(bvp_file):
        """Reads BVP CSV signal file (iBVP-style, simple like UBFC)."""
        df = pd.read_csv(bvp_file)
        if 'BVP' in df.columns:
            return df['BVP'].values
        elif len(df.columns) > 0:
            return df.iloc[:, 0].values
        raise ValueError(f"No BVP data found in {bvp_file}")
