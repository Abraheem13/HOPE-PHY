"""Continuum-memory block: a parameter group with an update-frequency class.

Each block is a residual MLP tagged with (level, period). Training uses
per-level learning rates (geometrically spaced, alpha_0 << ... << alpha_K);
at test time the TTT engine updates level k only every ``period_k`` steps.
This is the CMS primitive of Nested Learning instantiated for the PHY layer:
fast levels track instantaneous fading statistics, slow levels store
persistent scenario structure, and the slowest level carries an EMA anchor.
"""
from __future__ import annotations

import torch
from torch import nn


class MemoryBlock(nn.Module):
    def __init__(self, dim: int, level: int, period: int, hidden_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.level = level          # 0 = slowest ... K-1 = fastest
        self.period = period        # test-time update period (steps)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
