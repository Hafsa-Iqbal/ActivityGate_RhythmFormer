'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from evaluation.post_process import calculate_hr , calculate_psd

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss

class RhythmFormer_Loss(nn.Module):
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) :
           loss_time = 0

        # Epoch-ramp: time-domain loss dominates early (epoch 0-30), freq terms
        # are ramped up from 0.1× to 1.0× over the first 30 epochs.  This lets
        # the model first learn waveform correlation before chasing frequency peaks,
        # which is especially important for weak-signal inputs (e.g. thermal-only).
        if epoch >= 30:
            w_time, w_freq = 0.2, 1.0
        else:
            w_freq = 0.1 + 0.9 * (epoch / 30.0)   # 0.1 → 1.0
            w_time = 1.0 - 0.8 * (epoch / 30.0)   # 1.0 → 0.2

        loss = w_time * loss_time + w_freq * loss_CE + w_freq * loss_hr
        return loss

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        device = output.device
        k = k.type(torch.FloatTensor).to(device)
        two_pi_n_over_N = two_pi_n_over_N.to(device)
        hanning = hanning.to(device)
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).to(device=device, dtype=torch.float)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(40, 180, dtype=torch.float, device=device)
        #bpm_range = torch.arange(40, 260, dtype=torch.float, device=device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(40, 180, dtype=torch.float, device=device)
        #bpm_range = torch.arange(40, 260, dtype=torch.float, device=device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(40, 190, dtype=torch.float, device=device)
        #bpm_range = torch.arange(40, 180, dtype=torch.float, device=device)
        #bpm_range = torch.arange(40, 260, dtype=torch.float, device=device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(45, 150, dtype=torch.float, device=device)
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(device)
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)
    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        device = torch.device('cpu')
        pred_distribution = torch.Tensor(pred_distribution).to(device)
        target_distribution = torch.Tensor(target_distribution).to(device)
        return kl_loss(pred_distribution , target_distribution)

    @staticmethod
    def Scalar_Freq_Loss(inputs, gt_rate, Fs, rate_min, rate_max, std=2.0):
        """Frequency-domain CE + KL loss supervised by a scalar GT rate.

        Works for any periodic physiological signal:
          HR:  rate_min=45, rate_max=150 (bpm),   std=3.0
          BR:  rate_min=6,  rate_max=40  (br/min), std=2.0

        inputs:  (D,) predicted waveform tensor (on any device)
        gt_rate: float — ground-truth rate in the same units as rate_min/rate_max
        """
        device = inputs.device
        rate_range = torch.arange(rate_min, rate_max + 1, dtype=torch.float, device=device)
        ca = TorchLossComputer.complex_absolute(inputs.view(1, -1), Fs, rate_range)
        sa = ca / (torch.sum(ca) + 1e-8)

        gt_int = int(np.clip(gt_rate, rate_min, rate_max))
        target_dist = [normal_sampling(gt_int, i, std) for i in range(rate_min, rate_max + 1)]
        target_dist = torch.tensor([max(v, 1e-15) for v in target_dist],
                                   dtype=torch.float, device=device)

        gt_idx = torch.tensor([gt_int - rate_min], dtype=torch.long, device=device)
        loss_CE = F.cross_entropy(ca, gt_idx)
        loss_KL = kl_loss(sa, target_dist)
        return loss_CE + loss_KL

    @staticmethod
    def EDA_Loss(pred_signal, gt_eda, eda_log_mean=5.15, eda_log_std=1.56):
        """MSE between the mean of pred_signal and the log-normalised GT EDA value.

        pred_signal:  (D,) predicted EDA-proxy waveform (NOT z-scored — raw head output)
        gt_eda:       float — raw EDA value (any positive units)
        eda_log_mean: mean of log(EDA+1) over training set  (≈ 5.15 for SIM1)
        eda_log_std:  std  of log(EDA+1) over training set  (≈ 1.56 for SIM1)

        Log-transform makes EDA invariant to the wide dynamic range of EDA values.
        """
        gt_log  = math.log(max(gt_eda, 0.0) + 1.0)
        gt_norm = torch.tensor((gt_log - eda_log_mean) / eda_log_std,
                               dtype=torch.float, device=pred_signal.device)
        pred_mean = pred_signal.mean()
        return F.mse_loss(pred_mean, gt_norm)


class MultiTask_Loss(nn.Module):
    """Composite loss for multi-task RhythmFormerMT (HR + BR + EDA).

    HR and BR are supervised via frequency-domain CE+KL against scalar GT rates.
    EDA is supervised via scalar regression (log-normalised MSE on clip mean).
    """

    def __init__(self, tasks=('hr', 'br', 'eda'),
                 lambda_hr=1.0, lambda_br=1.0, lambda_eda=0.5,
                 Fs=8.5, eda_log_mean=5.15, eda_log_std=1.56):
        super().__init__()
        self.tasks        = tasks
        self.lam          = {'hr': lambda_hr, 'br': lambda_br, 'eda': lambda_eda}
        self.Fs           = Fs
        self.eda_log_mean = eda_log_mean
        self.eda_log_std  = eda_log_std

    def forward(self, preds, labels_scalar):
        """
        preds:         dict {task_name: (D,)} — per-sample waveform predictions
                       HR/BR should be z-scored; EDA should be raw head output.
        labels_scalar: (3,) float32 — [hr_bpm, br_bpm, eda_raw]
        """
        loss = torch.zeros(1, device=next(iter(preds.values())).device).squeeze()
        if 'hr' in preds:
            loss = loss + self.lam['hr'] * TorchLossComputer.Scalar_Freq_Loss(
                preds['hr'], labels_scalar[0].item(), self.Fs, 45, 150, std=3.0)
        if 'br' in preds:
            loss = loss + self.lam['br'] * TorchLossComputer.Scalar_Freq_Loss(
                preds['br'], labels_scalar[1].item(), self.Fs, 6, 40, std=2.0)
        if 'eda' in preds:
            loss = loss + self.lam['eda'] * TorchLossComputer.EDA_Loss(
                preds['eda'], labels_scalar[2].item(),
                self.eda_log_mean, self.eda_log_std)
        return loss