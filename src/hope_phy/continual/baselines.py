"""Continual-learning baselines for online adaptation under drift."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from ..metrics.nmse import nmse
from ..ttt.simple_adapter import _select_params


@dataclass
class BaselineConfig:
    lr: float = 6.4e-2
    inner_steps: int = 8
    adapt_scope: str = "all"
    grad_clip: float | None = 1.0
    ewc_lambda: float = 100.0
    fisher_decay: float = 0.95
    buffer_size: int = 128
    replay_batch: int = 8
    seed: int = 0


class _Base:
    def __init__(self, model: nn.Module, cfg: BaselineConfig):
        self.model, self.cfg = model, cfg
        self.params = _select_params(model, cfg.adapt_scope)
        self.t = 0

    def _step(self, loss: torch.Tensor) -> None:
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        gn = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item() \
            if any(g is not None for g in grads) else 0.0
        if gn == 0.0:
            return
        scale = 1.0
        if self.cfg.grad_clip is not None:
            scale = min(1.0, self.cfg.grad_clip / (gn + 1e-12))
        with torch.no_grad():
            for p, g in zip(self.params, grads):
                if g is not None:
                    p.add_(g, alpha=-self.cfg.lr * scale)


class NaiveFineTune(_Base):
    def __init__(self, model, cfg: BaselineConfig):
        cfg = BaselineConfig(**{**cfg.__dict__, "grad_clip": None})
        super().__init__(model, cfg)

    def observe(self, x, y):
        self.t += 1
        self.model.train()
        for _ in range(self.cfg.inner_steps):
            self._step(nmse(self.model(x), y))
        self.model.eval()
        return {"t": self.t}


class EWCAdapter(_Base):
    def __init__(self, model, cfg: BaselineConfig):
        super().__init__(model, cfg)
        self.anchor = [p.detach().clone() for p in self.params]
        self.fisher = [torch.zeros_like(p) for p in self.params]

    def observe(self, x, y):
        self.t += 1
        self.model.train()
        for _ in range(self.cfg.inner_steps):
            loss = nmse(self.model(x), y)
            pen = sum((f * (p - a) ** 2).sum()
                      for f, p, a in zip(self.fisher, self.params, self.anchor))
            self._step(loss + self.cfg.ewc_lambda * pen)
        loss = nmse(self.model(x), y)
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        with torch.no_grad():
            for f, g in zip(self.fisher, grads):
                if g is not None:
                    f.mul_(self.cfg.fisher_decay).add_(g.detach() ** 2,
                                                       alpha=1 - self.cfg.fisher_decay)
        self.model.eval()
        return {"t": self.t}


class ReplayAdapter(_Base):
    def __init__(self, model, cfg: BaselineConfig):
        super().__init__(model, cfg)
        self.buf = []
        self.rng = random.Random(cfg.seed)

    def _remember(self, x, y):
        if len(self.buf) < self.cfg.buffer_size:
            self.buf.append((x.detach().clone(), y.detach().clone()))
        else:
            j = self.rng.randint(0, self.t)
            if j < self.cfg.buffer_size:
                self.buf[j] = (x.detach().clone(), y.detach().clone())

    def observe(self, x, y):
        self.t += 1
        self.model.train()
        for _ in range(self.cfg.inner_steps):
            loss = nmse(self.model(x), y)
            if self.buf:
                k = min(self.cfg.replay_batch, len(self.buf))
                for xr, yr in self.rng.sample(self.buf, k):
                    loss = loss + nmse(self.model(xr), yr) / k
                loss = loss / 2.0
            self._step(loss)
        self._remember(x, y)
        self.model.eval()
        return {"t": self.t}


BASELINES = {"naive": NaiveFineTune, "ewc": EWCAdapter, "replay": ReplayAdapter}
