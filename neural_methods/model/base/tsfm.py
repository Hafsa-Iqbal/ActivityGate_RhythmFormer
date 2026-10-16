"""
Target Signal Constrained Factorization Module (TSFM).

Adapted from MMRPhys / FactorizePhys (PhysiologicAILab):
  https://github.com/PhysiologicAILab/MMRPhys

Core idea: NMF-based feature factorization where, during training, the
ground-truth BVP waveform (y_sig) acts as a temporal constraint on the
basis matrix, steering attention toward signal-relevant spatiotemporal
regions. At inference (y_sig=None) it falls back to unconstrained NMF.

Integration into RhythmFormer's video_BiFormerBlock:
  - TSFMGate  : standard BRA block + TSFM gate appended as a third sub-layer
  - TSFMOnly  : TSFM replaces the BRA attention sub-layer entirely
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSFM(nn.Module):
    """
    Target Signal Constrained Factorization Module.

    Operates on 5-D feature maps (B, C, T, H, W).

    Training (y_sig provided):
        Uses Signal-constrained NMF (SNMF, label mode).
        y_sig is resampled to the feature temporal length T, normalised to
        [1, 2], and used as element-wise temporal weights on the NMF bases.
        This biases the factorisation to weight time-steps that correspond
        to high BVP amplitude, focusing the model on physiologically
        relevant moments.

    Inference (y_sig=None):
        Gate is bypassed entirely — x is returned unchanged.
        This matches FactorizePhys MD_INFERENCE=False behaviour and
        prevents train/test feature-distribution mismatch that would
        otherwise arise from constrained vs. unconstrained NMF.

    Both paths run inside torch.no_grad() — NMF iterations are not
    differentiated. Gradients flow through pre_conv / post_conv and the
    element-wise gating (mul + InstanceNorm).

    Args:
        in_channels   : C of the input feature map.
        align_channels: Internal NMF channel width (dimensionality reduction).
        rank          : NMF rank R. Default 1 (as in MMRPhys).
        steps         : Number of NMF multiplicative update iterations.
        residual      : If True, add input skip after the gate (x + gate).
    """

    def __init__(
        self,
        in_channels: int,
        align_channels: int = 8,
        rank: int = 1,
        steps: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.steps = steps
        self.residual = residual
        self.align_channels = align_channels

        # Project to align_channels — ReLU ensures non-negativity for NMF
        self.pre_conv = nn.Sequential(
            nn.Conv3d(in_channels, align_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(align_channels),
            nn.ReLU(inplace=True),
        )

        # Project attention mask back to in_channels
        self.post_conv = nn.Sequential(
            nn.Conv3d(align_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels),
        )

        # Learnable bias for the element-wise gating (initialised to 1 so
        # gate ≈ identity at start)
        self.bias = nn.Parameter(torch.ones(1))

        # Instance norm on the gated output (affine=True to allow scale/shift)
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)

    # ------------------------------------------------------------------
    # NMF helpers (run inside no_grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _nmf_step(
        x: torch.Tensor,
        bases: torch.Tensor,
        coef: torch.Tensor,
        eps: float = 1e-8,
    ):
        """One Lee-Seung multiplicative NMF update.

        Args:
            x    : (B, D, N)  — feature matrix (D = temporal, N = spatial*ch)
            bases: (B, D, R)  — W (basis matrix)
            coef : (B, N, R)  — H (coefficient matrix)
        Returns:
            Updated (bases, coef).
        """
        # H update: H ← H ⊙ (X^T W) / (H W^T W)
        num_H = torch.bmm(x.transpose(1, 2), bases)           # (B, N, R)
        den_H = coef.bmm(bases.transpose(1, 2).bmm(bases))    # (B, N, R)
        coef = coef * (num_H / den_H.clamp(min=eps))
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)

        # W update: W ← W ⊙ (X H) / (W H^T H)
        num_W = torch.bmm(x, coef)                            # (B, D, R)
        den_W = bases.bmm(coef.transpose(1, 2).bmm(coef))     # (B, D, R)
        bases = bases * (num_W / den_W.clamp(min=eps))
        bases = torch.nan_to_num(bases, nan=0.0, posinf=1.0, neginf=0.0)

        return bases, coef

    @staticmethod
    def _snmf_step(
        x: torch.Tensor,
        bases: torch.Tensor,
        coef: torch.Tensor,
        y_weight: torch.Tensor,
        eps: float = 1e-8,
    ):
        """Signal-constrained NMF update (SNMF, label mode).

        Factorises X ≈ (y ⊙ W) H^T where y is the temporal BVP weight.
        The effective basis is W_eff = y ⊙ W.

        Args:
            x       : (B, D, N)
            bases   : (B, D, R)
            coef    : (B, N, R)
            y_weight: (B, D, 1)  — BVP signal normalised to [1, 2]
        """
        bases_eff = bases * y_weight                          # (B, D, R)

        # H update using signal-scaled bases
        num_H = torch.bmm(x.transpose(1, 2), bases_eff)
        den_H = coef.bmm(bases_eff.transpose(1, 2).bmm(bases_eff))
        coef = coef * (num_H / den_H.clamp(min=eps))
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)

        # W update: separate signal from basis so the base can generalise
        num_W = torch.bmm(x, coef)                           # (B, D, R)
        den_W = bases_eff.bmm(coef.transpose(1, 2).bmm(coef))
        bases = y_weight * (bases * (num_W / den_W.clamp(min=eps)))
        bases = torch.nan_to_num(bases, nan=0.0, posinf=1.0, neginf=0.0)

        return bases, coef

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y_sig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T, H, W) feature map.
            y_sig: (B, T_label) BVP waveform. Passed during training;
                   None at inference — gate is bypassed, x returned as-is.
        Returns:
            (B, C, T, H, W) feature map (same shape as x).
        """
        # Inference bypass: no y_sig → return x unchanged (identity).
        # Avoids train/test distribution mismatch from constrained vs.
        # unconstrained NMF producing different attention maps.
        if y_sig is None:
            return x

        B, C, T, H, W = x.shape

        # ---- Pre-conv + NMF reshape ----
        # pre_conv has ReLU → guaranteed non-negative for NMF
        x_proj = self.pre_conv(x)                            # (B, ac, T, H, W)
        N_sp = self.align_channels * H * W

        # Reshape to (B, D=T, N=ac*H*W)
        x_nmf = x_proj.permute(0, 2, 1, 3, 4).reshape(B, T, N_sp)

        # ---- Initialise NMF bases and coefficients ----
        bases = F.normalize(
            torch.ones(B, T, self.rank, device=x.device, dtype=x.dtype), dim=1
        )
        # Initial H via soft-assignment
        coef = F.softmax(
            torch.bmm(x_nmf.transpose(1, 2), bases), dim=-1  # (B, N, R)
        )

        # ---- Signal-constrained NMF (only reached during training) ----
        # Resample BVP from label length → feature temporal dim T
        with torch.no_grad():
            y_res = F.interpolate(
                y_sig.unsqueeze(1).to(dtype=x.dtype),
                size=T,
                mode='linear',
                align_corners=False,
            ).squeeze(1)                                     # (B, T)

            # Normalise to [1, 2] — always positive, avoids sign flips
            y_min = y_res.min(dim=1, keepdim=True)[0]
            y_max = y_res.max(dim=1, keepdim=True)[0]
            y_w = 1.0 + (y_res - y_min) / (y_max - y_min + 1e-8)
            y_w = y_w.unsqueeze(-1)                          # (B, T, 1)

            for _ in range(self.steps):
                bases, coef = self._snmf_step(x_nmf, bases, coef, y_w)

        # ---- Reconstruct attention mask ----
        att_nmf = torch.bmm(bases, coef.transpose(1, 2))    # (B, T, N_sp)
        att_nmf = (
            att_nmf
            .reshape(B, T, self.align_channels, H, W)
            .permute(0, 2, 1, 3, 4)                         # (B, ac, T, H, W)
        )

        # Project back to C channels
        att = self.post_conv(att_nmf)                        # (B, C, T, H, W)

        # ---- Element-wise gating ----
        # Shift both to positive, multiply, then normalise.
        # bias init=1 keeps gate ≈ identity at the start of training.
        x_gate = torch.mul(
            x   - x.min()   + self.bias,
            att - att.min() + self.bias,
        )
        x_gate = self.norm(x_gate)

        if self.residual:
            return x + x_gate
        return x_gate
