"""HOPE-PHY: backbone encoder -> Continuum Memory chain (+ Titans branch) -> head.

Architecture (v0)
-----------------
  x [B,P,D] --Linear+PE--> GRU/Transformer backbone --> h [B,d]
  h --> ContinuumMemory (K levels, slow->fast) --> z
  h --> TitansMemory retrieval --> m          (test-time-written context)
  concat(z, m) --> head --> y_hat [B,L,D]

Exposed for the TTT engine:
  * cms.param_groups(...)   -- per-level (lr, period) groups
  * titans.write(...)       -- surprise-gated memory write
  * cms.update_anchor / stabilise_to_anchor / reset_slow_to_anchor
"""
from __future__ import annotations

import torch
from torch import nn

from ..baselines.transformer import PositionalEncoding
from .continuum import ContinuumMemory
from .titans_memory import TitansMemory


class HopePhy(nn.Module):
    def __init__(self, feat_dim: int, l_fut: int, d_model: int = 256,
                 backbone: str = "gru", backbone_layers: int = 2,
                 cms_levels: int = 3, cms_base_period: int = 4,
                 use_titans: bool = True, titans_hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.l_fut, self.feat_dim = l_fut, feat_dim
        self.inp = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        if backbone == "gru":
            self.backbone = nn.GRU(d_model, d_model, backbone_layers,
                                   batch_first=True,
                                   dropout=dropout if backbone_layers > 1 else 0.0)
        else:
            enc = nn.TransformerEncoderLayer(d_model, 8, d_model * 2, dropout,
                                             batch_first=True, norm_first=True)
            self.backbone = nn.TransformerEncoder(enc, backbone_layers)
        self._backbone_kind = backbone

        self.cms = ContinuumMemory(d_model, cms_levels, cms_base_period,
                                   dropout=dropout)
        self.titans = TitansMemory(d_model, titans_hidden) if use_titans else None
        head_in = d_model * (2 if use_titans else 1)
        self.head = nn.Linear(head_in, l_fut * feat_dim)

    # ----- feature extraction (shared by forward and TTT writes) -----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.inp(x))
        if self._backbone_kind == "gru":
            out, _ = self.backbone(h)
        else:
            out = self.backbone(h)
        return out[:, -1]                              # [B, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        z = self.cms(h)
        if self.titans is not None:
            z = torch.cat([z, self.titans(h)], dim=-1)
        y = self.head(z)
        return y.view(x.shape[0], self.l_fut, self.feat_dim)

    # ----- multi-rate optimisation groups ----------------------------------
    def param_groups(self, lr_base: float, cms_lr_slow: float, cms_lr_ratio: float) -> list[dict]:
        backbone_and_io = [p for n, p in self.named_parameters()
                           if not n.startswith(("cms.", "titans."))]
        groups: list[dict] = [{"params": backbone_and_io, "lr": lr_base,
                               "level": -1, "period": 1}]
        groups += self.cms.param_groups(cms_lr_slow, cms_lr_ratio)
        if self.titans is not None:
            groups.append({"params": self.titans.parameters(), "lr": lr_base,
                           "level": -2, "period": 1})
        return groups
