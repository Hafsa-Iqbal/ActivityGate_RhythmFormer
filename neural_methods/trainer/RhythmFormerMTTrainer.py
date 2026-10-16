"""Trainer for RhythmFormerMT — multi-task (HR + BR + EDA) on SIM1 thermal."""

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from neural_methods.model.RhythmFormerMT import RhythmFormerMT
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import TorchLossComputer, MultiTask_Loss


def _normalise(sig):
    """Per-sample z-score normalisation along the last axis."""
    std = sig.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (sig - sig.mean(dim=-1, keepdim=True)) / std


class RhythmFormerMTTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device        = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir     = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size    = config.TRAIN.BATCH_SIZE
        self.config        = config
        self.min_valid_loss = None
        self.best_epoch    = 0

        cfg_mt  = config.MODEL.RHYTHMFORMERMT
        tasks   = list(cfg_mt.TASKS)
        Fs      = config.TRAIN.DATA.FS
        chunk   = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        # topk: match paper ratio (40/160 = 25% of frames).
        # Cap at min(D/4, 40) to preserve sparsity at low fps (8.5fps gives D=128,
        # old formula incorrectly gave topk=128 = full attention, defeating the mechanism).
        topk = min(chunk // 4, 40)

        fusion_type = getattr(cfg_mt, 'FUSION_TYPE', 'none')
        gate_blend_mode = getattr(cfg_mt, 'GATE_BLEND_MODE', 'learned')
        gate_placement = getattr(cfg_mt, 'GATE_PLACEMENT', 'stem')
        activity_window = getattr(cfg_mt, 'GATE_ACTIVITY_WINDOW', 0)

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RhythmFormerMT(
                tasks=tasks,
                in_channels=cfg_mt.IN_CHANNELS,
                topks=(topk, topk, topk),
                fusion_type=fusion_type,
                gate_blend_mode=gate_blend_mode,
                gate_placement=gate_placement,
                activity_window=activity_window,
            ).to(self.device)
            if getattr(config, 'NUM_OF_GPU_TRAIN', 0) > 0:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = MultiTask_Loss(
                tasks=tasks,
                Fs=Fs,
                eda_log_mean=getattr(cfg_mt, 'EDA_LOG_MEAN', 5.15),
                eda_log_std=getattr(cfg_mt,  'EDA_LOG_STD',  1.56),
            )
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)

        elif config.TOOLBOX_MODE == "only_test":
            self.model = RhythmFormerMT(
                tasks=tasks,
                in_channels=cfg_mt.IN_CHANNELS,
                topks=(topk, topk, topk),
                fusion_type=fusion_type,
                gate_blend_mode=gate_blend_mode,
                gate_placement=gate_placement,
                activity_window=activity_window,
            ).to(self.device)
            if getattr(config, 'NUM_OF_GPU_TRAIN', 0) > 0:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("RhythmFormerMTTrainer: unknown TOOLBOX_MODE")

        self.tasks = tasks
        self.Fs    = Fs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No train data")

        for epoch in range(self.max_epoch_num):
            print(f'\n====Training Epoch: {epoch}====')
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            for batch in tbar:
                tbar.set_description(f"Train epoch {epoch}")
                data   = batch[0].float().to(self.device)   # [N, D, C, H, W]
                labels = batch[1].float().to(self.device)   # [N, 3]

                N = data.shape[0]
                self.optimizer.zero_grad()

                preds = self.model(data, None)   # dict {task: [N, D]}

                # Z-score HR/BR (needed for freq-domain loss); leave EDA raw
                preds = {t: (_normalise(v) if t in ('hr', 'br') else v)
                         for t, v in preds.items()}

                loss = torch.zeros(1, device=self.device).squeeze()
                for ib in range(N):
                    per_sample = {t: preds[t][ib] for t in preds}
                    loss = loss + self.criterion(per_sample, labels[ib])
                loss = loss / N
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print(f'validation loss: {valid_loss:.4f}')
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f'Update best model! Best epoch: {self.best_epoch}')

        if not self.config.TEST.USE_LAST_EPOCH:
            print(f'Best epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss:.4f}')

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No valid data")
        print('\n===Validating===')
        losses = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch in vbar:
                vbar.set_description("Validation")
                data   = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)
                N = data.shape[0]
                preds = self.model(data, None)
                preds = {t: (_normalise(v) if t in ('hr', 'br') else v)
                         for t, v in preds.items()}
                for ib in range(N):
                    per_sample = {t: preds[t][ib] for t in preds}
                    l = self.criterion(per_sample, labels[ib])
                    losses.append(l.item())
                    vbar.set_postfix(loss=l.item())
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    # Test / evaluation
    # ------------------------------------------------------------------

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No test data")
        print('\n===Testing===')

        if not self.config.TEST.USE_LAST_EPOCH:
            self.load_best_model()

        self.model.eval()
        all_pred = {t: [] for t in self.tasks}
        all_gt   = {t: [] for t in self.tasks}
        task_cols = {'hr': 0, 'br': 1, 'eda': 2}

        all_waves = {t: [] for t in self.tasks if t in ('hr', 'br')}

        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=80):
                data   = batch[0].float().to(self.device)
                labels = batch[1].float()   # [N, 3] — keep on CPU for numpy ops
                preds  = self.model(data, None)

                for t in self.tasks:
                    col = task_cols[t]
                    gt_vals = labels[:, col].numpy()

                    if t in ('hr', 'br'):
                        # Extract dominant frequency from predicted waveform
                        rate_min = 45  if t == 'hr' else 6
                        rate_max = 150 if t == 'hr' else 40
                        wave_cpu = preds[t].cpu()
                        pred_vals = self._waveform_to_rate(wave_cpu, rate_min, rate_max)
                        # Store waveforms for SNR computation
                        all_waves[t].append(wave_cpu.numpy())
                    else:   # eda
                        cfg_mt    = self.config.MODEL.RHYTHMFORMERMT
                        log_mean  = getattr(cfg_mt, 'EDA_LOG_MEAN', 5.15)
                        log_std   = getattr(cfg_mt, 'EDA_LOG_STD',  1.56)
                        # Denormalise: pred_mean → log(EDA+1) → EDA
                        pred_log_norm = preds[t].mean(dim=-1).cpu().numpy()
                        pred_vals = np.exp(pred_log_norm * log_std + log_mean) - 1

                    all_pred[t].append(pred_vals)
                    all_gt[t].append(gt_vals)

        print('\n===Test Results===')
        for t in self.tasks:
            pred_arr = np.concatenate(all_pred[t])
            gt_arr   = np.concatenate(all_gt[t])
            mae  = np.mean(np.abs(pred_arr - gt_arr))
            rmse = np.sqrt(np.mean((pred_arr - gt_arr) ** 2))
            mape = np.mean(np.abs(pred_arr - gt_arr) / (np.abs(gt_arr) + 1e-8)) * 100
            unit = 'bpm' if t == 'hr' else ('br/min' if t == 'br' else 'raw')
            line = f'  {t.upper():3s}  MAE={mae:.3f} {unit}   RMSE={rmse:.3f}   MAPE={mape:.2f}%'
            if t in ('hr', 'br'):
                snr = self._compute_snr(all_waves[t], gt_arr, t)
                line += f'   SNR={snr:.3f} dB'
            print(line)

        # Save per-clip pred/GT for Bland-Altman plots
        save_dir = 'runs/exp/sim1_rgbt_rd_activitygate_test'
        os.makedirs(save_dir, exist_ok=True)
        for t in self.tasks:
            pred_arr = np.concatenate(all_pred[t])
            gt_arr   = np.concatenate(all_gt[t])
            np.save(os.path.join(save_dir, f'sim1_mt_pred_{t}.npy'), pred_arr)
            np.save(os.path.join(save_dir, f'sim1_mt_gt_{t}.npy'),   gt_arr)
        print(f'  Saved per-clip pred/GT to {save_dir}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_snr(self, wave_list, gt_rates, task):
        """Compute mean SNR of predicted waveforms using GT rate as signal frequency.

        wave_list : list of (N, D) numpy arrays collected per batch
        gt_rates  : (total_samples,) numpy array of GT rates (bpm or br/min)
        Returns mean SNR in dB across all samples.
        """
        waves = np.concatenate(wave_list, axis=0)   # (total_samples, D)
        snrs  = []
        for i, sig in enumerate(waves):
            hr_hz  = gt_rates[i] / 60.0
            N      = len(sig)
            freqs  = np.fft.rfftfreq(N, d=1.0 / self.Fs)
            psd    = np.abs(np.fft.rfft(sig)) ** 2
            if task == 'hr':
                band = (freqs >= 0.75) & (freqs <= 2.5)
            else:
                band = (freqs >= 0.1) & (freqs <= 0.67)
            sig_band = (freqs >= hr_hz - 0.1) & (freqs <= hr_hz + 0.1)
            p_total  = psd[band].sum()
            p_signal = psd[sig_band].sum()
            p_noise  = p_total - p_signal
            if p_noise > 0 and p_signal > 0:
                snrs.append(10 * np.log10(p_signal / p_noise))
        return float(np.mean(snrs)) if snrs else float('nan')

    def _waveform_to_rate(self, waveforms, rate_min, rate_max):
        """Extract dominant rate from a batch of waveforms via FFT.

        waveforms: (N, D) CPU tensor
        Returns:   (N,) numpy array of rates in the specified range.
        """
        rates = []
        bpm_range = torch.arange(rate_min, rate_max + 1, dtype=torch.float)
        for i in range(waveforms.shape[0]):
            sig = waveforms[i]
            ca  = TorchLossComputer.complex_absolute(sig.view(1, -1), self.Fs, bpm_range)
            idx = ca.argmax().item()
            rates.append(float(rate_min + idx))
        return np.array(rates)

    def load_best_model(self):
        # Prefer explicit MODEL_PATH from config (used in only_test mode)
        inference_path = getattr(self.config.INFERENCE, 'MODEL_PATH', '')
        if inference_path and os.path.exists(inference_path):
            model_path = inference_path
        else:
            model_path = os.path.join(
                self.model_dir,
                f'{self.model_file_name}_Epoch{self.best_epoch}.pth')
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            core = (self.model.module
                    if isinstance(self.model, torch.nn.DataParallel)
                    else self.model)
            core.load_state_dict(state)
            print(f'Loaded best model from {model_path}')
        else:
            print(f'[WARN] Best model not found at {model_path}, using last epoch.')

    def save_model(self, epoch):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(
            self.model_dir,
            f'{self.model_file_name}_Epoch{epoch}.pth')
        state = (self.model.module.state_dict()
                 if isinstance(self.model, torch.nn.DataParallel)
                 else self.model.state_dict())
        torch.save(state, model_path)
