"""Real cross-scenario streaming from LLM4CP test files (hard cut + gradual blend)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .llm4cp_dataset import LLM4CPDataset


@dataclass
class LLM4CPStreamCfg:
    uma_his: str = "data/llm4cp/Testing Dataset/H_U_his_test.mat"
    uma_pre: str = "data/llm4cp/Testing Dataset/H_U_pre_test.mat"
    umi_his: str = "data/llm4cp/Testing Dataset/Umi/H_U_his_test.mat"
    umi_pre: str = "data/llm4cp/Testing Dataset/Umi/H_U_pre_test.mat"
    per_speed_limit: int | None = 30
    shuffle_within: bool = True
    seed: int = 0


def _load_pair(cfg):
    uma = LLM4CPDataset(cfg.uma_his, cfg.uma_pre, per_speed_limit=cfg.per_speed_limit)
    umi = LLM4CPDataset(cfg.umi_his, cfg.umi_pre, per_speed_limit=cfg.per_speed_limit)
    return uma, umi


def make_llm4cp_stream(cfg: LLM4CPStreamCfg):
    rng = np.random.default_rng(cfg.seed)
    uma, umi = _load_pair(cfg)

    def order(ds):
        idx = np.arange(len(ds))
        if cfg.shuffle_within:
            rng.shuffle(idx)
        return idx

    for i in order(uma):
        xh, yf = uma[int(i)]
        yield xh.unsqueeze(0), yf.unsqueeze(0), "uma"
    for i in order(umi):
        xh, yf = umi[int(i)]
        yield xh.unsqueeze(0), yf.unsqueeze(0), "umi"


@dataclass
class BlendedStreamCfg(LLM4CPStreamCfg):
    bands: tuple = ((0.0, 150), (0.25, 150), (0.5, 150), (0.75, 150), (1.0, 150))


def make_blended_stream(cfg: BlendedStreamCfg):
    rng = np.random.default_rng(cfg.seed)
    uma, umi = _load_pair(cfg)
    n = min(len(uma), len(umi))
    order = np.arange(n)
    if cfg.shuffle_within:
        rng.shuffle(order)

    ptr = 0
    for beta, count in cfg.bands:
        tag = f"b{beta:.2f}"
        for _ in range(count):
            i = int(order[ptr % n]); ptr += 1
            xa, ya = uma[i]; xb, yb = umi[i]
            xh = (1 - beta) * xa + beta * xb
            yf = (1 - beta) * ya + beta * yb
            p = xh.pow(2).mean().sqrt().clamp_min(1e-6)
            xh, yf = xh / p, yf / p
            yield xh.unsqueeze(0), yf.unsqueeze(0), tag


@dataclass
class MultiTimescaleStreamCfg(LLM4CPStreamCfg):
    n_steps: int = 900
    jitter_amp: float = 0.35
    jitter_period: int = 5
    n_report_bands: int = 5


def make_multitimescale_stream(cfg: MultiTimescaleStreamCfg):
    rng = np.random.default_rng(cfg.seed)
    uma, umi = _load_pair(cfg)
    n = min(len(uma), len(umi))
    order = np.arange(n)
    rng.shuffle(order)

    jitter = 0.0
    for t in range(cfg.n_steps):
        slow = t / max(cfg.n_steps - 1, 1)
        if t % cfg.jitter_period == 0:
            jitter = rng.uniform(-cfg.jitter_amp, cfg.jitter_amp)
        beta = float(np.clip(slow + jitter, 0.0, 1.0))
        i = int(order[t % n])
        xa, ya = uma[i]; xb, yb = umi[i]
        xh = (1 - beta) * xa + beta * xb
        yf = (1 - beta) * ya + beta * yb
        p = xh.pow(2).mean().sqrt().clamp_min(1e-6)
        xh, yf = xh / p, yf / p
        band = min(int(slow * cfg.n_report_bands), cfg.n_report_bands - 1)
        yield xh.unsqueeze(0), yf.unsqueeze(0), f"s{band}"
