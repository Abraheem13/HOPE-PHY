"""Loader for the LLM4CP released QuaDRiGa dataset (PKU-PCNI/LLM4CP)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import PowerNormalizer, complex_to_real


def _load_mat_array(path: Path, key: str | None = None) -> np.ndarray:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            k = key or max(keys, key=lambda kk: int(np.prod(f[kk].shape)))
            arr = np.array(f[k])
            if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
                arr = arr["real"] + 1j * arr["imag"]
            return arr
    except OSError:
        from scipy.io import loadmat
        d = {kk: v for kk, v in loadmat(path).items() if not kk.startswith("__")}
        k = key or max(d, key=lambda kk: d[kk].size)
        return d[k]


def _canonicalize(arr, t_len, n_blocks, n_speeds):
    shape = list(arr.shape)

    def _find(size, used):
        for i, s in enumerate(shape):
            if s == size and i not in used:
                return i
        raise ValueError(f"axis of size {size} not found in {shape}")

    used = []
    ax_block = _find(n_blocks, used); used.append(ax_block)
    ax_speed = _find(n_speeds, used); used.append(ax_speed)
    ax_time = _find(t_len, used); used.append(ax_time)
    rest = [i for i in range(arr.ndim) if i not in used]
    # CRITICAL: h5py reverses MATLAB axis order, and the his/pre files are
    # stored differently. If this array is reverse-stored (block axis appears
    # AFTER the time axis), its trailing feature axes are also reversed, so we
    # flip them to a common (subcarrier, Nh, Nv, pol) order. Without this the
    # input and target feature vectors index different subcarriers/antennas.
    if ax_block > ax_time:
        rest = rest[::-1]
    arr = np.transpose(arr, [ax_block, ax_speed, ax_time, *rest])
    return arr.reshape(n_blocks, n_speeds, t_len, -1)


class LLM4CPDataset(Dataset):
    P_LEN, L_LEN = 16, 4
    N_BLOCKS, N_SPEEDS = 900, 10

    def __init__(self, his_path, pre_path, normalize: bool = True,
                 per_speed_limit: int | None = None,
                 n_blocks: int | None = None, n_speeds: int | None = None):
        nb = n_blocks or self.N_BLOCKS
        ns = n_speeds or self.N_SPEEDS
        his = _load_mat_array(Path(his_path))
        pre = _load_mat_array(Path(pre_path))

        if nb not in his.shape:
            cand = [s for s in his.shape if s not in (self.P_LEN, 48, 4, 2)]
            nb = max(cand)

        his = _canonicalize(his, self.P_LEN, nb, ns)
        pre = _canonicalize(pre, self.L_LEN, nb, ns)
        b = min(his.shape[0], pre.shape[0])
        if per_speed_limit:
            b = min(b, per_speed_limit)
        his, pre = his[:b], pre[:b]

        his = complex_to_real(his).astype(np.float32)
        pre = complex_to_real(pre).astype(np.float32)
        B, S = his.shape[0], his.shape[1]
        self.speed_index = np.tile(np.arange(S), B)
        his = his.reshape(B * S, self.P_LEN, -1)
        pre = pre.reshape(B * S, self.L_LEN, -1)

        norm = PowerNormalizer() if normalize else (lambda a, c: (a, c))
        self.samples = [norm(h, p) for h, p in zip(his, pre)]
        self.feat_dim = self.samples[0][0].shape[-1]
        self.n_speeds = S

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        xh, yf = self.samples[i]
        return (torch.from_numpy(np.ascontiguousarray(xh)),
                torch.from_numpy(np.ascontiguousarray(yf)))
