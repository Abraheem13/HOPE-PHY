"""Minimal online adapter -- the honest method matching the ablation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..metrics.nmse import nmse


@dataclass
class SimpleAdapterConfig:
    enabled: bool = True
    lr: float = 1e-3
    inner_steps: int = 8
    grad_clip: float = 1.0
    trust_grad_norm: float = 1e4
    adapt_scope: str = "head"


def _select_params(model: nn.Module, scope: str) -> list[nn.Parameter]:
    if scope == "all":
        return [p for p in model.parameters() if p.requires_grad]
    named = dict(model.named_parameters())
    if scope == "head":
        return [p for n, p in named.items() if n.startswith("head.")]
    if scope == "head+last":
        keep = []
        for n, p in named.items():
            if n.startswith("head."):
                keep.append(p)
            elif ".layers." in n and n.split(".layers.")[1].split(".")[0].isdigit():
                keep.append(p)
        return keep or [p for n, p in named.items() if n.startswith("head.")]
    raise ValueError(f"unknown scope {scope}")


class SimpleOnlineAdapter:
    def __init__(self, model: nn.Module, cfg: SimpleAdapterConfig):
        self.model, self.cfg = model, cfg
        self.params = _select_params(model, cfg.adapt_scope)
        self.t = 0

    def observe(self, x_hist: torch.Tensor, y_true: torch.Tensor) -> dict:
        if not self.cfg.enabled:
            return {"t": self.t, "surprise": None}
        self.t += 1
        self.model.train()
        surprise0 = None
        for _ in range(self.cfg.inner_steps):
            pred = self.model(x_hist)
            loss = nmse(pred, y_true)
            if surprise0 is None:
                surprise0 = loss.detach().item()
            grads = torch.autograd.grad(loss, self.params, allow_unused=True)
            gn = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item() \
                if any(g is not None for g in grads) else 0.0
            if gn == 0.0 or gn > self.cfg.trust_grad_norm:
                break
            scale = min(1.0, self.cfg.grad_clip / (gn + 1e-12))
            with torch.no_grad():
                for p, g in zip(self.params, grads):
                    if g is not None:
                        p.add_(g, alpha=-self.cfg.lr * scale)
        self.model.eval()
        return {"t": self.t, "surprise": surprise0}
