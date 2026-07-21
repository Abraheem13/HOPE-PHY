from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import PowerNormalizer, complex_to_real


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            out = {}
            for k in f.keys():
                arr = np.array(f[k])
                if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
                    arr = arr["real"] + 1j * arr["imag"]
                out[k] = arr
            return out
    except OSError:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def _pick_key(d: dict[str, np.ndarray], pinned: str | None) -> str:
    if pinned:
        if pinned not in d:
            raise KeyError(f"Pinned key '{pinned}' not in file; found {list(d)}")
        return pinned
    key = max(d, key=lambda k: d[k].size)
    print(f"[llm4cp] auto-detected data key '{key}' shape={d[key].shape} "
          f"dtype={d[key].dtype} -- PIN THIS in configs/data/llm4cp.yaml")
    return key


class LLM4CPDataset(Dataset):
    def __init__(self, mat_path: str | Path, p_hist: int = 16, l_fut: int = 4,
                 data_key: str | None = None, normalize: bool = True,
                 limit: int | None = None):
        d = _load_mat(Path(mat_path))
        arr = d[_pick_key(d, data_key)]
        if arr.ndim < 2:
            raise ValueError(f"Unexpected array ndim={arr.ndim}")
        if arr.shape[0] < arr.shape[-1] and arr.ndim >= 3:
            arr = np.transpose(arr, list(range(arr.ndim))[::-1])
        n, t = arr.shape[0], arr.shape[1]
        if t < p_hist + l_fut:
            raise ValueError(f"time axis {t} < P+L={p_hist + l_fut}; check permutation")
        arr = arr.reshape(n, t, -1)
        if np.iscomplexobj(arr):
            arr = complex_to_real(arr)
        arr = arr.astype(np.float32)
        if limit:
            arr = arr[:limit]
        norm = PowerNormalizer() if normalize else (lambda a, b: (a, b))
        self.samples = [norm(s[:p_hist], s[p_hist:p_hist + l_fut]) for s in arr]
        self.feat_dim = self.samples[0][0].shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        xh, yf = self.samples[i]
        return torch.from_numpy(np.ascontiguousarray(xh)), torch.from_numpy(np.ascontiguousarray(yf))
