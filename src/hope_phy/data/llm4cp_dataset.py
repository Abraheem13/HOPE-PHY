"""Loader for the LLM4CP released QuaDRiGa dataset (PKU-PCNI/LLM4CP).

Input and target live in SEPARATE .mat files (per readme.txt):
  H_U_his_{split}: [900,10,16,48,4,4,2]  historical uplink (P=16, input)
  H_U_pre_{split}: [900,10, 4,48,4,4,2]  future uplink     (L=4, TDD target)
  H_D_pre_{split}: [900,10, 4,48,4,4,2]  future downlink   (L=4, FDD target)

Logical axes: [block, speed, time, subcarrier(48), Nh(4), Nv/pol(4), pol(2)].
D = 48*4*4*2 = 1536 complex -> 3072 real (Re/Im stacked).

h5py transposes MATLAB arrays and the two files use different orders, so we
locate the time axis by its known length (16 vs 4), fold (block,speed) into the
sample axis, flatten the rest into D. Robust to either storage order.
"""
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


def _canonicalize(arr: np.ndarray, t_len: int, n_blocks: int, n_speeds: int) -> np.ndarray:
    """Reshape arbitrary-order LLM4CP array to canonical [N, T, D] complex."""
    shape = list(arr.shape)

    def _find(size, used):
        for i, s in enumerate(shape):
            if s == size and i not in used:
                return i
        raise ValueError(f"axis of size {size} not found in {shape}")

    used: list[int] = []
    ax_block = _find(n_blocks, used); used.append(ax_block)
    ax_speed = _find(n_speeds, used); used.append(ax_speed)
    ax_time = _find(t_len, used); used.append(ax_time)
    rest = [i for i in range(arr.ndim) if i not in used]
    arr = np.transpose(arr, [ax_block, ax_speed, ax_time, *rest])
    return arr.reshape(n_blocks * n_speeds, t_len, -1)


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

        # test files: 100 blocks/speed; auto-detect the block axis size.
        if nb not in his.shape:
            cand = [s for s in his.shape if s not in (self.P_LEN, 48, 4, 2)]
            nb = max(cand)

        his = _canonicalize(his, self.P_LEN, nb, ns)   # [N,16,D]
        pre = _canonicalize(pre, self.L_LEN, nb, ns)   # [N, 4,D]
        assert his.shape[0] == pre.shape[0], (his.shape, pre.shape)
        assert his.shape[2] == pre.shape[2], (his.shape, pre.shape)

        his = complex_to_real(his).astype(np.float32)
        pre = complex_to_real(pre).astype(np.float32)

        if per_speed_limit:
            keep = np.arange(min(per_speed_limit * ns, his.shape[0]))
            his, pre = his[keep], pre[keep]

        norm = PowerNormalizer() if normalize else (lambda a, b: (a, b))
        self.samples = [norm(h, p) for h, p in zip(his, pre)]
        self.feat_dim = self.samples[0][0].shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        xh, yf = self.samples[i]
        return (torch.from_numpy(np.ascontiguousarray(xh)),
                torch.from_numpy(np.ascontiguousarray(yf)))
