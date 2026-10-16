# ActivityGate-RhythmFormer

Code for "Thermal-Guided Activity Gating for Multi-Modal Remote Physiological Measurement" — a multi-modal RGB-Thermal extension of RhythmFormer for remote photoplethysmography (rPPG) estimation, introducing an unsupervised ActivityGate spatial gating module and a multi-task extension for simultaneous heart rate (HR) and breathing rate (BR) estimation.

This repository is anonymized for double-blind review. Author and institutional details will be added upon publication.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Training and evaluation configs are under `configs/train_configs/`. For example, to train on the iBVP dataset with the proposed RGBT ActivityGate model:

```bash
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_ActivityGate.yaml
```

Dataset paths in the config files are placeholders (`/path/to/workspace/...`) — update `DATA_PATH` and `CACHED_PATH` in the relevant config to point to your local copy of the iBVP or SIM1 dataset before running.

Face detection defaults to Haar Cascade (`BACKEND: 'HC'`, bundled with OpenCV — no extra download needed), which is what all reported results use. An alternative YOLOv5Face backend (`BACKEND: 'Y5F'`) is also implemented but requires its own pretrained checkpoint, not included in this repository.

## Repository structure

- `neural_methods/` — model architecture (RhythmFormer backbone, ActivityGate module, loss functions, trainers)
- `dataset/` — data loaders for iBVP, SIM1, and other supported datasets
- `evaluation/` — metrics and post-processing (HR/BR extraction from predicted waveforms)
- `configs/` — training and inference configuration files

## License

This project builds on [RhythmFormer](https://github.com/zizheng-guo/RhythmFormer) (MIT License; see `LICENSE.txt`), with additional modules and extensions described in the paper.
