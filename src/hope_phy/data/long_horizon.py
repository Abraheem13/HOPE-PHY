"""Long-horizon dataset: P=8 -> L=8 re-windowed from the 16-frame his files.

At this horizon (4 ms) persistence collapses (+1.5 dB), so prediction must use
learned dynamics -- the regime where cross-scenario transfer can genuinely break.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from .llm4cp_dataset import _load_mat_array, _canonicalize


class LongHorizonDataset(Dataset):
    P_LEN, L_LEN = 8, 8

    def __init__(self, his_path, per_speed_limit=None, limit=None):
        a = _load_mat_array(Path(his_path))
        nb = [s for s in a.shape if s not in (16, 48, 4, 2) and s != 10][0]
        c = _canonicalize(a, 16, nb, 10)              # [B,S,16,D] complex
        del a
        if per_speed_limit:
            c = c[:per_speed_limit]
        B, S = c.shape[0], c.shape[1]
        c = c.reshape(B * S, 16, c.shape[-1]).astype(np.complex64)
        self.speed_index = np.tile(np.arange(S), B)
        X = np.concatenate([c.real, c.imag], -1).astype(np.float32)  # [N,16,2D]
        del c
        if limit:
            X = X[:limit]; self.speed_index = self.speed_index[:limit]
        hist, fut = X[:, :self.P_LEN], X[:, self.P_LEN:]
        p = np.sqrt((hist ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-6
        self.hist = hist / p
        self.fut = fut / p
        self.feat_dim = self.hist.shape[-1]
        self.n_speeds = S

    def __len__(self):
        return self.hist.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.hist[i].copy()),
                torch.from_numpy(self.fut[i].copy()))
