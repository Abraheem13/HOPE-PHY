from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from .synthetic import SyntheticCfg, generate_series
from .transforms import PowerNormalizer, complex_to_real, window_series


@dataclass
class SegmentSpec:
    tag: str
    t_steps: int = 600
    severity: float = 1.0
    doppler_max: float = 0.08
    shadow_sigma: float = 0.10


@dataclass
class StreamCfg:
    segments: list[SegmentSpec] = field(default_factory=lambda: [
        SegmentSpec("uma_like", severity=0.8, doppler_max=0.05),
        SegmentSpec("umi_like", severity=1.5, doppler_max=0.10),
        SegmentSpec("highway_like", severity=2.0, doppler_max=0.20),
    ])
    n_rb: int = 12
    n_ant: int = 4
    p_hist: int = 16
    l_fut: int = 4
    seed: int = 0


def make_stream(cfg: StreamCfg):
    rng = np.random.default_rng(cfg.seed)
    norm = PowerNormalizer()
    for seg in cfg.segments:
        scfg = SyntheticCfg(n_series=1, t_steps=seg.t_steps, n_rb=cfg.n_rb,
                            n_ant=cfg.n_ant, severity=seg.severity,
                            doppler_max=seg.doppler_max,
                            shadow_sigma=seg.shadow_sigma,
                            p_hist=cfg.p_hist, l_fut=cfg.l_fut)
        series = complex_to_real(generate_series(scfg, rng))
        for xh, yf in window_series(series, cfg.p_hist, cfg.l_fut, stride=cfg.l_fut):
            xh, yf = norm(xh, yf)
            yield (torch.from_numpy(xh.copy()).unsqueeze(0),
                   torch.from_numpy(yf.copy()).unsqueeze(0), seg.tag)
