"""RhythmFormer: Multi-Modal rPPG Estimation (RGB / RGB+Thermal / Thermal-Only)

block_type controls which attention block is used inside TPT stages:
  "standard"  : original video_BiFormerBlock (baseline).
  "tsfm_gate" : BRA + TSFM gate as a third sub-layer.
  "tsfm_only" : TSFM replaces BRA entirely.
"""
from typing import Optional, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, Union
from timm.models.layers import trunc_normal_
from neural_methods.model.base.video_bra import (
    video_BiFormerBlock,
    video_BiFormerBlock_TSFMGate,
    video_BiFormerBlock_TSFMOnly,
)

_BLOCK_REGISTRY = {
    "standard":  video_BiFormerBlock,
    "tsfm_gate": video_BiFormerBlock_TSFMGate,
    "tsfm_only": video_BiFormerBlock_TSFMOnly,
}

class ThermalSkinMask(nn.Module):
    """Proposed A: Single-conv + sigmoid converts downsampled thermal to a
    spatial soft-mask (N*D, 1, H/4, W/4) that is broadcast over all 64
    feature channels of the Fusion_Stem output.

    Input : (N*D, 1, H/4, W/4)  — thermal channel after 4× avg-pool
    Output: (N*D, 1, H/4, W/4)  — values in (0, 1)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, thermal_ds):
        return torch.sigmoid(self.bn(self.conv(thermal_ds)))


class SoftThermalAttention(nn.Module):
    """Proposed B: Two-conv branch converts downsampled thermal to a
    per-channel-spatial attention map (N*D, 64, H/4, W/4), providing
    richer guidance than the single-channel skin mask.

    Input : (N*D, 1,  H/4, W/4)  — thermal channel after 4× avg-pool
    Output: (N*D, 64, H/4, W/4)  — values in (0, 1)
    """
    def __init__(self, out_channels=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, thermal_ds):
        return self.branch(thermal_ds)


class AnatomicalThermalGate(nn.Module):
    """Proposed D: Anatomically-anchored thermal gate.

    Blends a learned thermal mask with a fixed anatomical prior (Gaussian
    heatmap over superficial vessel locations at 32×32) then applies a
    residual modulation gate so the network can elect to ignore thermal.

      self.alpha : blend scalar, init=0.0 → sigmoid=0.5 (equal blend at start)
      self.beta  : gate strength, init=-4.0 → sigmoid≈0.018 (near-identity at start)
      self.prior : fixed buffer (1,1,32,32), NOT a learnable parameter

    Input : thermal_ds (N*D, 1, 32, 32)
    Output: gate       (N*D, 1, 32, 32), values in (1-β, 1), broadcasts over 64ch
    """
    def __init__(self):
        super().__init__()
        # Fix 3: 2-channel input — static thermal frame + temporal diff magnitude
        self.conv  = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn    = nn.BatchNorm2d(1)
        self.alpha = nn.Parameter(torch.tensor(0.0))   # blend: sigmoid(0)=0.5
        # Fix 1: beta=-1 → sigmoid≈0.27 → gate range (0.73,1.0) from epoch 1
        self.beta  = nn.Parameter(torch.tensor(-1.0))  # gate strength: sigmoid(-1)≈0.27
        self.register_buffer("prior", self._build_anatomical_prior())

    @staticmethod
    def _build_anatomical_prior() -> torch.Tensor:
        """(1,1,32,32) Gaussian heatmap over superficial vessel regions.
        Grid: row 0 = top, col 0 = left (32×32 = 128×128 face crop downsampled 4×).
          Forehead          : center (r=4,  c=16), σ=(4, 14)
          Glabella          : center (r=10, c=16), σ=(2,  3)
          Temporal artery L : center (r=14, c=3),  σ=(6,  2)
          Temporal artery R : center (r=14, c=29), σ=(6,  2)
          Nasolabial fold L : center (r=20, c=9),  σ=(3,  2)   # Fix 2
          Nasolabial fold R : center (r=20, c=23), σ=(3,  2)   # Fix 2
          Chin              : center (r=27, c=16), σ=(2,  5)   # Fix 2
        """
        H, W = 32, 32
        rows = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
        cols = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)

        def gauss2d(r0, c0, sr, sc):
            return torch.exp(-0.5 * (((rows - r0) / sr) ** 2 + ((cols - c0) / sc) ** 2))

        prior = torch.max(torch.stack([
            gauss2d(4,  16, 4,  14),   # forehead
            gauss2d(10, 16, 2,   3),   # glabella
            gauss2d(14,  3, 6,   2),   # temporal artery L
            gauss2d(14, 29, 6,   2),   # temporal artery R
            gauss2d(20,  9, 3,   2),   # nasolabial fold L
            gauss2d(20, 23, 3,   2),   # nasolabial fold R
            gauss2d(27, 16, 2,   5),   # chin
        ], dim=0), dim=0).values

        prior = prior / (prior.max() + 1e-8)
        return prior.unsqueeze(0).unsqueeze(0)   # (1, 1, 32, 32)

    def forward(self, thermal_ds: torch.Tensor, thermal_diff: torch.Tensor) -> torch.Tensor:
        # Fix 3: static frame + temporal diff magnitude → gate sees WHERE T is changing
        gate_input   = torch.cat([thermal_ds, thermal_diff], dim=1)           # (N*D,2,32,32)
        thermal_mask = torch.sigmoid(self.bn(self.conv(gate_input)))          # (N*D,1,32,32)
        alpha        = torch.sigmoid(self.alpha)                              # scalar
        combined_map = alpha * thermal_mask + (1.0 - alpha) * self.prior     # (N*D,1,32,32)
        beta         = torch.sigmoid(self.beta)                               # scalar ≈0.27
        gate         = 1.0 - beta + beta * combined_map                      # residual gate
        return gate


class ActivityMapGate(nn.Module):
    """Proposed F: Data-driven thermal activity gate.

    Replaces the fixed anatomical prior with a per-clip thermal activity map:
        activity_map = mean |T_t - T_{t-1}| over all D frames in the clip.

    This is subject-specific and frame-position-agnostic — it highlights
    exactly where the thermal camera captures pulsatile signal for each
    individual, regardless of face size, crop margin, or head pose.

    Blends:
        combined = alpha * learned_mask(frame_t, diff_t) + (1-alpha) * activity_map
        gate     = 1 - beta + beta * combined               (residual, near-identity init)

    Input:
        thermal_ds   : (N*D, 1, 32, 32) — current frame thermal, downsampled
        activity_ds  : (N*D, 1, 32, 32) — per-clip activity map, expanded to N*D frames
        thermal_diff : (N*D, 1, 32, 32) — instantaneous |T_t - T_{t-1}|
    Output:
        gate         : (N*D, 1, 32, 32), values in (1-beta, 1)

    blend_mode:
        "learned"       — alpha is a learnable parameter (default, proposed method).
        "activity_only" — alpha fixed so sigmoid(alpha)≈0: gate driven solely by activity_ds.
        "mask_only"     — alpha fixed so sigmoid(alpha)≈1: gate driven solely by thermal_mask.
    """
    def __init__(self, blend_mode: str = "learned"):
        super().__init__()
        # 2-channel input: current frame + instantaneous diff
        self.conv  = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn    = nn.BatchNorm2d(1)
        self.blend_mode = blend_mode
        if blend_mode == "learned":
            self.alpha = nn.Parameter(torch.tensor(0.0))     # blend: sigmoid(0)=0.5
        elif blend_mode == "activity_only":
            self.register_buffer("alpha", torch.tensor(-10.0))  # sigmoid(-10)≈0
        elif blend_mode == "mask_only":
            self.register_buffer("alpha", torch.tensor(10.0))   # sigmoid(10)≈1
        else:
            raise ValueError(f"Unknown blend_mode '{blend_mode}'. "
                              f"Choose from 'learned', 'activity_only', 'mask_only'.")
        self.beta = nn.Parameter(torch.tensor(-1.0))  # gate strength: sigmoid(-1)≈0.27

    def forward(self, thermal_ds: torch.Tensor,
                activity_ds: torch.Tensor,
                thermal_diff: torch.Tensor) -> torch.Tensor:
        gate_input   = torch.cat([thermal_ds, thermal_diff], dim=1)    # (N*D,2,32,32)
        thermal_mask = torch.sigmoid(self.bn(self.conv(gate_input)))   # (N*D,1,32,32)
        alpha        = torch.sigmoid(self.alpha)
        combined_map = alpha * thermal_mask + (1.0 - alpha) * activity_ds
        beta         = torch.sigmoid(self.beta)
        gate         = 1.0 - beta + beta * combined_map
        return gate


class Fusion_Stem(nn.Module):
    """Parameterized Fusion Stem supporting variable input channels.

    For C input channels:
      - Main branch conv: C -> 64
      - Diff branch conv: 4*C -> 64  (4 temporal differences x C channels)
    Supports: C=3 (RGB), C=4 (RGBT), C=1 (Thermal-only).

    fusion_type controls optional post-stem thermal guidance (RGBT only):
      "concat"   — original behaviour, no thermal guidance module.
      "skin_mask" — ThermalSkinMask broadcast over 64 channels.
      "soft_attn" — SoftThermalAttention per channel per spatial location.
    """
    def __init__(self, in_channels=3, apha=0.5, belta=0.5, fusion_type="concat",
                 gate_blend_mode="learned", gate_placement="stem", activity_window=0):
        super(Fusion_Stem, self).__init__()
        diff_channels = 4 * in_channels  # 4 temporal diffs x C channels
        self.gate_placement = gate_placement
        self.activity_window = activity_window if activity_window and activity_window > 0 else None

        # Main branch: C-channel input
        self.stem11 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        # Diff branch: (4 * C) channels
        self.stem12 = nn.Sequential(
            nn.Conv2d(diff_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        # Fusion conv layers
        self.stem21 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.apha = apha
        self.belta = belta
        self.fusion_type = fusion_type

        # Thermal guidance modules (only instantiated when needed)
        if fusion_type in ("skin_mask", "soft_attn", "anatomical_gate", "activity_gate"):
            # AvgPool to downsample thermal from H×W to H/4×W/4
            # (matches Fusion_Stem spatial output: stride-2 conv + stride-2 maxpool)
            self.thermal_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            if fusion_type == "skin_mask":
                self.thermal_guide = ThermalSkinMask()
            elif fusion_type == "anatomical_gate":
                self.thermal_guide = AnatomicalThermalGate()
            elif fusion_type == "activity_gate":
                self.thermal_guide = ActivityMapGate(blend_mode=gate_blend_mode)
            else:
                self.thermal_guide = SoftThermalAttention(out_channels=64)

        elif fusion_type == "separate_stem":
            # Proposed E: Decoupled RGB and thermal stems — thermal never contaminates
            # RGB feature learning. AnatomicalThermalGate is applied to thermal
            # features BEFORE they are concatenated with RGB features.
            #
            # Channel split: RGB 3→48, Thermal 1→16  (sum=64, ratio mirrors 3:1 channels)
            # Diff split:    RGB diffs 12→48, Thermal diffs 4→16
            self.thermal_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.thermal_guide = AnatomicalThermalGate()

            self.rgb_stem11 = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.thm_stem11 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.rgb_stem12 = nn.Sequential(
                nn.Conv2d(12, 48, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.thm_stem12 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        input_x = x               # saved for thermal extraction (before reshape)
        N, D, C, H, W = x.shape  # [N, D, C, H, W]

        # Bidirectional multi-order temporal differences (per-sample, no batch contamination)
        # 5-frame window: t-2, t-1, t, t+1, t+2 with boundary replication
        x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D-2, :, :, :]], 1)  # shift right 2
        x2 = torch.cat([x[:, :1, :, :, :], x[:, :D-1, :, :, :]], 1)                       # shift right 1
        x3 = x                                                                               # center
        x4 = torch.cat([x[:, 1:, :, :, :], x[:, D-1:, :, :, :]], 1)                       # shift left 1
        x5 = torch.cat([x[:, 2:, :, :, :], x[:, D-1:, :, :, :], x[:, D-1:, :, :, :]], 1) # shift left 2

        # --- Proposed E: decoupled RGB / thermal stems ---
        if self.fusion_type == "separate_stem":
            x3_flat  = x3.contiguous().view(N * D, C, H, W)
            rgb_flat = x3_flat[:, :3]   # (N*D, 3, H, W)
            thm_flat = x3_flat[:, 3:]   # (N*D, 1, H, W)

            # Anatomical gate from static thermal frame
            thm_ds = self.thermal_pool(thm_flat)   # (N*D, 1, H/4, W/4)
            gate   = self.thermal_guide(thm_ds)    # (N*D, 1, H/4, W/4)

            # Main path: separate stems, gate thermal before concat
            rgb_feat = self.rgb_stem11(rgb_flat)         # (N*D, 48, H/4, W/4)
            thm_feat = self.thm_stem11(thm_flat) * gate  # (N*D, 16, H/4, W/4)
            x        = torch.cat([rgb_feat, thm_feat], dim=1)  # (N*D, 64, H/4, W/4)

            # Diff path: split diffs by modality, gate thermal diffs with same gate
            diffs_flat    = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 4 * C, H, W)
            rgb_diff_feat = self.rgb_stem12(diffs_flat[:, :12])          # (N*D, 48, H/4, W/4)
            thm_diff_feat = self.thm_stem12(diffs_flat[:, 12:]) * gate   # (N*D, 16, H/4, W/4)
            x_diff        = torch.cat([rgb_diff_feat, thm_diff_feat], dim=1)  # (N*D, 64, H/4, W/4)

            # Shared fusion layers (stem21/stem22 unchanged)
            x_path1 = self.apha * x + self.belta * x_diff
            x_path1 = self.stem21(x_path1)
            x_path2 = self.stem22(x_diff)
            return (self.apha * x_path1 + self.belta * x_path2), None

        # --- Original shared-stem path (concat / skin_mask / soft_attn / anatomical_gate) ---
        # 4 temporal differences concatenated along channel dim → 4 × C channels
        x_diff = self.stem12(
            torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 4 * C, H, W)
        )

        # Main path: process original frames
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        # Fusion layer 1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        # Fusion layer 2
        x_path2 = self.stem22(x_diff)
        x = self.apha * x_path1 + self.belta * x_path2

        # --- Proposed A / B / D / F: thermal guidance applied after stem fusion ---
        if self.fusion_type in ("skin_mask", "soft_attn", "anatomical_gate", "activity_gate"):
            # Extract thermal channel (last channel, index -1) from raw input
            # input_x: (N, D, C, H, W) → thermal: (N*D, 1, H, W)
            thermal_raw  = input_x[:, :, -1:, :, :]                           # (N, D, 1, H, W)
            thermal_flat = thermal_raw.contiguous().view(N * D, 1, H, W)
            thermal_ds   = self.thermal_pool(thermal_flat)                     # (N*D, 1, H/4, W/4)
            # Instantaneous temporal diff for anatomical_gate and activity_gate
            thermal_prev    = torch.cat([thermal_raw[:, :1], thermal_raw[:, :-1]], dim=1)
            thermal_diff_ds = self.thermal_pool(
                (thermal_raw - thermal_prev).abs().contiguous().view(N * D, 1, H, W)
            )                                                                  # (N*D, 1, H/4, W/4)
            if self.fusion_type == "anatomical_gate":
                guide = self.thermal_guide(thermal_ds, thermal_diff_ds)
            elif self.fusion_type == "activity_gate":
                # Mean |ΔT| activity map — per-pixel mean absolute frame diff over the clip.
                # Highlights where thermal signal changes most; computed clip-wide so it
                # captures pulsatility region for each subject regardless of face alignment.
                # Shape: (N, D, 1, H, W) → mean over D (or a centered sub-window) → (N, 1, H, W)
                thermal_diffs = (thermal_raw - torch.cat(
                    [thermal_raw[:, :1], thermal_raw[:, :-1]], dim=1
                )).abs()                                                        # (N, D, 1, H, W)
                if self.activity_window is not None and self.activity_window < D:
                    w = self.activity_window
                    start = (D - w) // 2
                    activity_hw = thermal_diffs[:, start:start + w].mean(dim=1)  # (N, 1, H, W)
                else:
                    activity_hw = thermal_diffs.mean(dim=1)                        # (N, 1, H, W)
                activity_ds  = self.thermal_pool(activity_hw)                   # (N, 1, H/4, W/4)
                # Normalize per-sample to [0,1]
                activity_ds  = activity_ds / (
                    activity_ds.amax(dim=(2, 3), keepdim=True) + 1e-8
                )
                # Expand from (N,1,32,32) → (N*D,1,32,32) to match frame-level tensors
                activity_ds  = activity_ds.unsqueeze(1).expand(N, D, 1, H // 4, W // 4) \
                                          .contiguous().view(N * D, 1, H // 4, W // 4)
                guide = self.thermal_guide(thermal_ds, activity_ds, thermal_diff_ds)
            else:
                guide = self.thermal_guide(thermal_ds)                         # (N*D, 1|64, H/4, W/4)

            if self.fusion_type == "activity_gate" and self.gate_placement != "stem":
                # Defer application to a later point in the backbone (see RhythmFormer._get_features)
                return x, guide
            x = x * guide                                                      # broadcast if guide is 1ch

        return x, None

class TPT_Block(nn.Module):
    """Hierarchical Temporal Periodic Transformer Block"""
    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5,
                 block_class=video_BiFormerBlock,
                 tsfm_kwargs: Dict[str, Any] | None = None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch, 2))

        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            )
            self.downsample_layers.append(downsample_layer)

            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
                nn.ELU(),
            )
            self.upsample_layers.append(upsample_layer)

        extra = tsfm_kwargs or {}
        self.blocks = nn.ModuleList([
            block_class(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                num_heads=num_heads,
                t_patch=t_patch,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_dwconv=side_dwconv,
                **extra,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, y_sig=None):
        for i in range(self.layer_n):
            x = self.downsample_layers[i](x)

        for blk in self.blocks:
            x = blk(x, y_sig)

        for i in range(self.layer_n):
            x = self.upsample_layers[i](x)

        return x

class RhythmFormer(nn.Module):
    """Multi-Modal RhythmFormer supporting RGB (3ch), RGBT (4ch), and Thermal (1ch).

    Args:
        block_type  : "standard" | "tsfm_gate" | "tsfm_only"
                      Controls the inner block used in every TPT stage.
        tsfm_kwargs : Dict of TSFM hyper-parameters forwarded to the block
                      constructors when block_type != "standard".
                      Keys: tsfm_align_ch, tsfm_rank, tsfm_steps, tsfm_residual.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        in_channels: int = 3,
        dim: int = 64,
        frame: int = 160,
        image_size: Optional[Tuple[int, int, int]] = (160, 128, 128),
        in_chans: int = 64,
        head_dim: int = 16,
        stage_n: int = 3,
        embed_dim: list = [64, 64, 64],
        mlp_ratios: list = [1.5, 1.5, 1.5],
        depth: list = [2, 2, 2],
        t_patchs: Union[int, Tuple[int]] = (2, 4, 8),
        topks: Union[int, Tuple[int]] = (40, 40, 40),
        side_dwconv: int = 3,
        drop_path_rate: float = 0.,
        use_checkpoint_stages: list = [],
        block_type: str = "standard",
        tsfm_kwargs: Dict[str, Any] | None = None,
        fusion_type: str = "concat",
        gate_blend_mode: str = "learned",
        gate_placement: str = "stem",
        activity_window: int = 0,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.stage_n = stage_n
        self.block_type = block_type
        self.gate_placement = gate_placement
        self.embed_dim = embed_dim

        self.Fusion_Stem = Fusion_Stem(in_channels=in_channels, apha=0.5, belta=0.5,
                                       fusion_type=fusion_type, gate_blend_mode=gate_blend_mode,
                                       gate_placement=gate_placement, activity_window=activity_window)
        self.patch_embedding = nn.Conv3d(
            in_chans, embed_dim[0],
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4)
        )

        block_class = _BLOCK_REGISTRY.get(block_type)
        if block_class is None:
            raise ValueError(
                f"Unknown block_type '{block_type}'. "
                f"Choose from {list(_BLOCK_REGISTRY.keys())}."
            )

        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in embed_dim]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        for i in range(stage_n):
            stage = TPT_Block(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i],
                mlp_ratio=mlp_ratios[i],
                drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                t_patch=t_patchs[i],
                topk=topks[i],
                side_dwconv=side_dwconv,
                block_class=block_class,
                tsfm_kwargs=tsfm_kwargs,
            )
            self.stages.append(stage)

        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1, stride=1, padding=0)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _get_features(self, x, y_sig=None):
        """Run backbone up to (but not including) the final projection head.

        Returns:
            features_last: [N, embed_dim[-1], D]  — temporal feature tensor.
        """
        N, D, C, H, W = x.shape
        x, guide = self.Fusion_Stem(x)
        x = x.view(N, D, 64, H//4, W//4).permute(0, 2, 1, 3, 4)
        x = self.patch_embedding(x)

        guide_resized = None
        if guide is not None:
            # guide: (N*D, 1, H/4, W/4) → further 4x downsample to match patch-embedded
            # spatial resolution (H/16, W/16), then reshape to broadcast over (N, C, D, H/16, W/16).
            guide_small = F.avg_pool2d(guide, kernel_size=4, stride=4)          # (N*D,1,H/16,W/16)
            guide_resized = guide_small.view(N, D, 1, H // 16, W // 16).permute(0, 2, 1, 3, 4)
            if self.gate_placement == "post_patch_embed":
                x = x * guide_resized

        for i in range(self.stage_n):
            x = self.stages[i](x, y_sig)
            if guide_resized is not None and self.gate_placement == "between_stages":
                x = x * guide_resized
        features_last = torch.mean(x, 3)
        features_last = torch.mean(features_last, 3)  # [N, C, D]
        return features_last

    def forward(self, x, y_sig=None):
        """
        Args:
            x    : [N, D, C, H, W]  — video clip, C = in_channels.
            y_sig: [N, D]           — BVP waveform used by TSFM blocks during
                                      training.  Ignored by standard blocks and
                                      at inference (None).
        Returns:
            rPPG: [N, T]  — predicted rPPG waveform.
        """
        features_last = self._get_features(x, y_sig)  # [N, C, D]

        # rPPG prediction
        rPPG = self.ConvBlockLast(features_last).squeeze(1)  # [N, T_model]
        """
        # Pad/trim to match the input length
        if rPPG.shape[1] != orig_D:
            if rPPG.shape[1] < orig_D:
                # Pad with zeros
                pad_length = orig_D - rPPG.shape[1]
                rPPG = F.pad(rPPG, (0, pad_length))
            else:
                # Trim excess
                rPPG = rPPG[:, :orig_D]
        
        assert rPPG.shape[1] == orig_D, f"Output length {rPPG.shape[1]} != input {orig_D}" """
        return rPPG  # Guaranteed [N,180]

"""
if __name__ == "__main__":
    model = RhythmFormer(frame=180)
    x = torch.randn(2, 180, 4, 128, 128)
    output = model(x)
    print(f" Input: [2,180,4,128,128] and Output: {output.shape} ✓")
"""

class RhythmFormer_LateFusion(nn.Module):
    """Proposed C: Late Fusion — separate RGB and Thermal encoders fused at
    prediction level with a learned softmax-weighted sum.

    RGB branch    : full-width RhythmFormer (embed_dim=64, in_channels=3).
    Thermal branch: half-width RhythmFormer (embed_dim=32, in_channels=1).
                    Always uses standard blocks (no TSFM) — thermal has no
                    cardiac signal to steer factorisation.
    Fusion        : softmax(fusion_logits)[0] * pred_rgb
                    + softmax(fusion_logits)[1] * pred_thermal
                    Initialised to [2.0, -2.0] so the model starts heavily
                    weighted towards RGB (w_rgb ≈ 0.98, w_thermal ≈ 0.02).
    """

    def __init__(
        self,
        frame: int = 160,
        image_size: Optional[Tuple[int, int, int]] = (160, 128, 128),
        drop_path_rate: float = 0.,
        drop_rate: float = 0.,
        block_type: str = "standard",
        tsfm_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__()

        # Full-width RGB branch
        self.rgb_branch = RhythmFormer(
            in_channels=3,
            dim=64,
            frame=frame,
            image_size=image_size,
            in_chans=64,
            embed_dim=[64, 64, 64],
            drop_path_rate=drop_path_rate,
            block_type=block_type,
            tsfm_kwargs=tsfm_kwargs,
            fusion_type="concat",
        )

        # Half-width Thermal branch (always standard blocks)
        self.thermal_branch = RhythmFormer(
            in_channels=1,
            dim=32,
            frame=frame,
            image_size=image_size,
            in_chans=64,          # Fusion_Stem always outputs 64ch
            embed_dim=[32, 32, 32],
            drop_path_rate=drop_path_rate,
            block_type="standard",
            tsfm_kwargs=None,
            fusion_type="concat",
        )

        # Learnable fusion weights: init heavily towards RGB
        self.fusion_logits = nn.Parameter(torch.tensor([2.0, -2.0]))

    def forward(self, x, y_sig=None):
        """
        Args:
            x    : (N, D, 4, H, W) — RGBT input; channels 0-2 = RGB, channel 3 = Thermal.
            y_sig: (N, D) or None  — BVP signal for TSFM blocks in RGB branch.
        Returns:
            rPPG : (N, D) — fused prediction.
        """
        rgb     = x[:, :, :3, :, :]   # (N, D, 3, H, W)
        thermal = x[:, :, 3:, :, :]   # (N, D, 1, H, W)

        pred_rgb     = self.rgb_branch(rgb, y_sig)  # (N, D)
        pred_thermal = self.thermal_branch(thermal)  # (N, D) — no y_sig

        w = torch.softmax(self.fusion_logits, dim=0)
        return w[0] * pred_rgb + w[1] * pred_thermal


##### last forward block and forward block of fusion stem where manually tensor is created of specific length