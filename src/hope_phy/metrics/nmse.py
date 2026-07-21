"""NMSE metrics -- the standard channel-prediction figure of merit.

NMSE = E[ ||H_hat - H||^2 / ||H||^2 ],   reported as 10*log10(NMSE) dB.
Computed per-sample then averaged (matches LLM4CP/Transformer-JSAC protocol).
"""
from __future__ import annotations

import torch


def nmse(pred: torch.Tensor, target: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """pred/target: [B, L, D] (real-stacked). Per-sample NMSE, mean if reduce."""
    err = (pred - target).flatten(1).pow(2).sum(-1)
    ref = target.flatten(1).pow(2).sum(-1).clamp_min(1e-12)
    val = err / ref
    return val.mean() if reduce else val


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(10.0 * torch.log10(nmse(pred, target).clamp_min(1e-12)))


class StreamingNMSE:
    """Accumulates per-step NMSE over a stream; supports segment tagging.

    Used by the streaming scenario-transition protocol to report, per segment:
    steady-state NMSE, post-transition degradation depth, and recovery time
    (steps until NMSE returns within ``tol_db`` of the segment steady state).
    """

    def __init__(self):
        self.steps: list[float] = []
        self.tags: list[str] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, tag: str = "") -> None:
        self.steps.append(float(nmse(pred, target)))
        self.tags.append(tag)

    def db_series(self) -> list[float]:
        import math

        return [10.0 * math.log10(max(v, 1e-12)) for v in self.steps]

    def segment_summary(self, tol_db: float = 1.0) -> dict[str, dict[str, float]]:
        import math

        out: dict[str, dict[str, float]] = {}
        series = self.db_series()
        # Steady state per tag = mean of last half of that tag's steps.
        by_tag: dict[str, list[int]] = {}
        for i, t in enumerate(self.tags):
            by_tag.setdefault(t, []).append(i)
        for tag, idx in by_tag.items():
            vals = [series[i] for i in idx]
            half = vals[len(vals) // 2 :] or vals
            steady = sum(half) / len(half)
            peak = max(vals[: max(len(vals) // 4, 1)])
            rec = next((k for k, v in enumerate(vals) if v <= steady + tol_db), len(vals))
            out[tag] = {"steady_db": steady, "peak_db": peak,
                        "degradation_db": peak - steady, "recovery_steps": float(rec)}
        return out
