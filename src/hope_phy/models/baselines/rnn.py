"""Recurrent baselines (RNN / LSTM / GRU).

Canonical interface for every predictor in this repo:
    forward(x_hist [B, P, D]) -> y_hat [B, L, D]
"""
from __future__ import annotations

import torch
from torch import nn

_CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RecurrentPredictor(nn.Module):
    def __init__(self, feat_dim: int, l_fut: int, cell: str = "lstm",
                 hidden: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.l_fut = l_fut
        self.encoder = _CELLS[cell](feat_dim, hidden, layers,
                                    batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, l_fut * feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        y = self.head(out[:, -1])
        return y.view(x.shape[0], self.l_fut, self.feat_dim)
