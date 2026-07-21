from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import PowerNormalizer, complex_to_real, window_series


@dataclass
class SyntheticCfg:
    n_series: int = 64
    t_steps: int = 400
    n_rb: int = 12
    n_ant: int = 4
    n_clusters: int = 6
    doppler_max: float = 0.08
    shadow_sigma: float = 0.10
    shadow_period: int = 15
    regime_period: int = 100
    severity: float = 1.0
    p_hist: int = 16
    l_fut: int = 4
    seed: int = 0


def generate_series(cfg: SyntheticCfg, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(cfg.t_steps)[:, None]
    d = cfg.n_rb * cfg.n_ant
    h = np.zeros((cfg.t_steps, d), dtype=np.complex64)
    n_regimes = cfg.t_steps // cfg.regime_period + 1
    for r in range(n_regimes):
        lo, hi = r * cfg.regime_period, min((r + 1) * cfg.regime_period, cfg.t_steps)
        if lo >= hi:
            break
        gains = rng.rayleigh(0.5 + cfg.severity * rng.random(), (cfg.n_clusters, d))
        f_d = cfg.doppler_max * rng.uniform(-1, 1, (cfg.n_clusters, 1))
        phi = rng.uniform(0, 2 * np.pi, (cfg.n_clusters, d))
        ramp = rng.uniform(0, 2 * np.pi, (1, d)) * np.ones((cfg.n_clusters, 1))
        seg = np.zeros((hi - lo, d), dtype=np.complex64)
        tt = t[lo:hi]
        for c in range(cfg.n_clusters):
            seg += (gains[c] * np.exp(1j * (2 * np.pi * f_d[c] * tt + phi[c] + ramp[c]))).astype(np.complex64)
        h[lo:hi] = seg / np.sqrt(cfg.n_clusters)
    n_blocks = cfg.t_steps // cfg.shadow_period + 1
    walk = np.cumsum(rng.normal(0, cfg.shadow_sigma, n_blocks))
    g = np.exp(np.repeat(walk, cfg.shadow_period)[: cfg.t_steps])[:, None]
    return (h * g).astype(np.complex64)


class SyntheticChannelDataset(Dataset):
    def __init__(self, cfg: SyntheticCfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        norm = PowerNormalizer()
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(cfg.n_series):
            series = complex_to_real(generate_series(cfg, rng))
            for xh, yf in window_series(series, cfg.p_hist, cfg.l_fut, stride=cfg.l_fut):
                self.samples.append(norm(xh, yf))
        self.feat_dim = self.samples[0][0].shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        xh, yf = self.samples[i]
        return torch.from_numpy(xh.copy()), torch.from_numpy(yf.copy())


def make_synthetic(cfg_dict: dict) -> SyntheticChannelDataset:
    keys = {f.name for f in SyntheticCfg.__dataclass_fields__.values()}
    return SyntheticChannelDataset(SyntheticCfg(**{k: v for k, v in cfg_dict.items() if k in keys}))
