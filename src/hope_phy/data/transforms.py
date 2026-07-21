from __future__ import annotations
import numpy as np
import torch


def complex_to_real(h: np.ndarray) -> np.ndarray:
    return np.concatenate([h.real, h.imag], axis=-1).astype(np.float32)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    f = x.shape[-1] // 2
    return torch.complex(x[..., :f], x[..., f:])


class PowerNormalizer:
    def __call__(self, x_hist: np.ndarray, y_fut: np.ndarray):
        p = np.sqrt(np.mean(np.abs(x_hist) ** 2) + 1e-12)
        return x_hist / p, y_fut / p


def window_series(h: np.ndarray, p: int, l: int, stride: int = 1):
    t = h.shape[0]
    out = []
    for s in range(0, t - p - l + 1, stride):
        out.append((h[s : s + p], h[s + p : s + p + l]))
    return out
