"""Transformer channel predictor (encoder-only, Jiang et al. JSAC'22 style)."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerPredictor(nn.Module):
    def __init__(self, feat_dim: int, l_fut: int, d_model: int = 256,
                 nhead: int = 8, layers: int = 4, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.l_fut, self.feat_dim = l_fut, feat_dim
        self.inp = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, l_fut * feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.inp(x)))
        y = self.head(h[:, -1]).view(x.shape[0], self.l_fut, self.feat_dim)
        # residual prediction: anchor on the last observed frame so the model
        # only learns the DELTA. At low mobility the optimal delta is ~0, so
        # this inherits the strong persistence baseline for free.
        return x[:, -1:].detach() + y
