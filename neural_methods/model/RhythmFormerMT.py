"""RhythmFormerMT — multi-task extension of RhythmFormer.

Shares the full RhythmFormer backbone and replaces the single BVP output head
with up to three independent 1-D convolutional heads for:
  'hr'  — heart rate proxy waveform  (same as original BVP head)
  'br'  — breathing rate proxy waveform
  'eda' — EDA / sudomotor proxy waveform

Each head is a single Conv1d(embed_dim[-1], 1, kernel_size=1).
The forward pass returns a dict {task_name: (N, D) tensor}.
"""

import torch
import torch.nn as nn
from neural_methods.model.RhythmFormer import RhythmFormer


class RhythmFormerMT(RhythmFormer):
    """Multi-task RhythmFormer with separate output heads per physiological signal."""

    VALID_TASKS = ('hr', 'br', 'eda')

    def __init__(self, tasks=('hr', 'br', 'eda'), **kwargs):
        """
        Args:
            tasks : tuple/list of task names from {'hr', 'br', 'eda'}.
            **kwargs: forwarded to RhythmFormer.__init__
                      (in_channels, frame_depth, drop_rate, fusion_type, …).
        """
        super().__init__(**kwargs)
        self.tasks = tuple(tasks)
        feat_dim = self.embed_dim[-1]   # 64

        # Remove the single-task head inherited from RhythmFormer
        del self.ConvBlockLast

        self.head_hr  = nn.Conv1d(feat_dim, 1, kernel_size=1) if 'hr'  in self.tasks else None
        self.head_br  = nn.Conv1d(feat_dim, 1, kernel_size=1) if 'br'  in self.tasks else None
        self.head_eda = nn.Conv1d(feat_dim, 1, kernel_size=1) if 'eda' in self.tasks else None

        # Initialise new heads
        for head in (self.head_hr, self.head_br, self.head_eda):
            if head is not None:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, x, y_sig=None):
        """
        Args:
            x    : [N, D, C, H, W]
            y_sig: [N, D] or None  (TSFM signal; pass None for SIM1/scalar supervision)

        Returns:
            dict {task_name: (N, D)} for each active task.
        """
        features = self._get_features(x, y_sig)   # [N, feat_dim, D]

        out = {}
        if self.head_hr  is not None:
            out['hr']  = self.head_hr(features).squeeze(1)   # [N, D]
        if self.head_br  is not None:
            out['br']  = self.head_br(features).squeeze(1)
        if self.head_eda is not None:
            out['eda'] = self.head_eda(features).squeeze(1)
        return out
