# ActivityGate-RhythmFormer

## Setup

```bash
pip install -r requirements.txt
```

## Usage

See [RUNNING.md](RUNNING.md) for full step-by-step instructions (dataset preparation, reproducing the main results, and the ablation studies).

Training and evaluation configs are under `configs/train_configs/`. For example, to train on the iBVP dataset with the proposed RGBT ActivityGate model use the follwoign:

```bash
python main.py --config_file configs/train_configs/intra/5IBVP_RGBT_ActivityGate.yaml
```

Dataset paths in the config files (`DATA_PATH`, `CACHED_PATH`) are set to the original development machine's local paths, update them in the relevant config to point to your own local copy of the iBVP or SIM1 dataset before running.

Face detection defaults to Haar Cascade (`BACKEND: 'HC'`, bundled with OpenCV and no extra download needed), which is what all reported results use. An alternative YOLOv5Face backend (`BACKEND: 'Y5F'`) is also implemented but requires its own pretrained checkpoint.

## Repository structure

- `neural_methods/`: model architecture (RhythmFormer backbone, ActivityGate module, loss functions, trainers)
- `dataset/`: data loaders for iBVP, SIM1, and other supported datasets
- `evaluation/`: metrics and post-processing (HR/BR extraction from predicted waveforms)
- `configs/`: training and inference configuration files

## License

This project builds on [RhythmFormer](https://github.com/zizheng-guo/RhythmFormer) (MIT License; see `LICENSE.txt`), with additional modules and extensions described in the paper.

The optional YOLOv5Face backend (`dataset/data_loader/face_detector/`) is adapted from [yolov5-face](https://github.com/deepcam-cn/yolov5-face) and is licensed separately under GPL-3.0 (see `dataset/data_loader/face_detector/LICENSE`). It is not used by the default configuration or any reported results.
