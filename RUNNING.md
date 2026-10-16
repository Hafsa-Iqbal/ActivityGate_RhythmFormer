# Step-by-Step Guide
 
This guide covers environment setup, dataset preparation, and reproducing the paper's main results (iBVP and SIM1, with and without ActivityGate) and key ablations.

## 1. Environment setup

```bash
git clone <this-repo-url>
cd ActivityGate_RhythmFormer
pip install -r requirements.txt
```

Python 3.10 is recommended (matches the environment used for reported results).

## 2. Dataset preparation

You need local copies of the **iBVP** and/or **SIM1** datasets.

- **iBVP**: Expected layout:
  ```
  <iBVP_root>/
    p02_a/
      rgb/*.bmp        (timestamped RGB frames)
      thermal/*.raw    (timestamped 16-bit thermal frames)
      p02_a_bvp.csv    (ground-truth BVP waveform)
    p07_a/
      ...
  ```
 

- **SIM1**: Expected layout:
  ```
  <SIM1_root>/
    Raw Thermal Data/T001/T001-RD.dat, T001-RD.inf, ...
    Structured Study Data/extracted/T001/.../T001-*.avi1.avi
    R-Friendly Study Data/T001.csv
  ```

## 3. Point the configs at your data

Every config under `configs/train_configs/intra/` has `DATA_PATH` and `CACHED_PATH` fields (under `TRAIN.DATA`, `VALID.DATA`, `TEST.DATA`) set to the paths used during development. Update these three occurrences in whichever config you run to point at your own `<iBVP_root>` / `<SIM1_root>`, and to a writable cache directory (preprocessed clips are written there on first run and reused afterward).

`DO_PREPROCESS` controls whether raw data is re-chunked into clips on this run:
- `True`: preprocess from raw data (required the first time, or after any preprocessing-affecting config change).
- `False`: reuse an existing cache at `CACHED_PATH` (faster; use once the cache already exists).

## 4. Reproduce the main results

**iBVP, proposed model (RGBT with ActivityGate):**
```bash
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_ActivityGate.yaml
```

**iBVP, no-gate baseline:**
```bash
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_NoGate.yaml
```

**SIM1, proposed model (RGBT with ActivityGate, multi-task HR+BR):**
```bash
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_ActivityGate.yaml
```

**SIM1, no-gate baseline:**
```bash
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_NoGate.yaml
```

Each run trains for the number of epochs set in the config, evaluates on the held-out test split at the end, and prints MAE/RMSE/MAPE/Pearson/SNR (iBVP) or per-task MAE/RMSE/MAPE/SNR (SIM1). Checkpoints are saved under `runs/exp/<run-name>/PreTrainedModels/`.

## 5. Reproduce the ablation studies

Component-wise ablation (isolating the ActivityGate's two internal signals and the temporal alignment strategy):

```bash
# Activity-map-only gate (learned mask disabled)
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_ActivityGate_ActivityOnly.yaml
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_ActivityGate_ActivityOnly.yaml

# Mask-only gate (activity map disabled)
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_ActivityGate_MaskOnly.yaml
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_ActivityGate_MaskOnly.yaml

# Naive temporal alignment (index-based pairing instead of nearest-timestamp / fps-ratio)
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_NaiveAlign_ActivityGate.yaml
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_NaiveAlign_ActivityGate.yaml
```

Gate-placement and activity-window ablations follow the same pattern (`*_PlacementPostPatch.yaml`, `*_PlacementBetweenStages.yaml`, `*_Window25/50/75.yaml`, for both datasets).

## 6. Evaluate an existing checkpoint (no retraining)

Set `TOOLBOX_MODE: "only_test"` and point `INFERENCE.MODEL_PATH` at a saved checkpoint — see `configs/train_configs/intra/SIM1_RGBT_RD_ActivityGate_test_only.yaml` for a worked example. Then run the same way:

```bash
python main.py --config_file configs/train_configs/intra/SIM1_RGBT_RD_ActivityGate_test_only.yaml
```

## 7. Switching face-detection backend (optional)

All files uses Haar Cascade (`CROP_FACE.BACKEND: 'HC'`, bundled with OpenCV) by default. To use the alternative YOLOv5Face backend instead, set `BACKEND: 'Y5F'` in the relevant config's `CROP_FACE` block, its pretrained checkpoint is included under `dataset/data_loader/face_detector/ckpts/` (see `dataset/data_loader/face_detector/LICENSE`).
