"""Test-time adaptation (TTA) engine for HOPE-PHY -- v2 (adapts enough to bite)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..metrics.nmse import nmse
from ..models.cms.hope_phy import HopePhy


@dataclass
class TTTConfig:
    enabled: bool = True
    lr_slow: float = 1e-3
    lr_ratio: float = 8.0
    inner_steps: int = 8
    med_period: int = 4
    slow_period: int = 16
    grad_clip: float = 1.0
    trust_grad_norm: float = 1e4
    anchor_decay: float = 0.99
    anchor_beta: float = 0.95
    reset_patience: int = 40
    reset_surprise: float = 3.0
    uniform_periods: bool = False
    freeze_backbone: bool = False


class TTTEngine:
    def __init__(self, model: HopePhy, cfg: TTTConfig):
        self.model, self.cfg = model, cfg
        self.t = 0
        self._bad = 0
        self.log: list[dict] = []

        named = dict(model.named_parameters())

        def pick(pred):
            return [p for n, p in named.items() if pred(n)]

        n_lvl = model.cms.n_levels
        fast_lvl, slow_lvl = n_lvl - 1, 0
        med_lvls = [l for l in range(n_lvl) if l not in (fast_lvl, slow_lvl)]

        fast_params = pick(lambda n: n.startswith("head.")) \
            + pick(lambda n: n.startswith(f"cms.blocks.{fast_lvl}."))
        if model.titans is not None:
            fast_params += pick(lambda n: n.startswith("titans.key_proj") or
                                          n.startswith("titans.val_proj"))
        med_params = []
        for l in med_lvls:
            med_params += pick(lambda n, l=l: n.startswith(f"cms.blocks.{l}."))
        slow_params = pick(lambda n: n.startswith(f"cms.blocks.{slow_lvl}."))
        if not cfg.freeze_backbone:
            slow_params += pick(lambda n: n.startswith(("inp.", "backbone.")))

        up = cfg.uniform_periods
        self.groups = [
            {"name": "fast", "params": fast_params,
             "lr": cfg.lr_slow * (cfg.lr_ratio ** 2), "period": 1},
            {"name": "med", "params": med_params or [torch.nn.Parameter(torch.zeros(1))],
             "lr": cfg.lr_slow * cfg.lr_ratio, "period": 1 if up else cfg.med_period},
            {"name": "slow", "params": slow_params,
             "lr": cfg.lr_slow, "period": 1 if up else cfg.slow_period},
        ]

    def observe(self, x_hist: torch.Tensor, y_true: torch.Tensor) -> dict:
        if not self.cfg.enabled:
            return {"t": self.t, "surprise": None, "updated": []}
        self.t += 1
        model, cfg = self.model, self.cfg
        model.train()

        if model.titans is not None:
            with torch.no_grad():
                feat = model.encode(x_hist)
            model.titans.write(feat, feat)

        surprise0 = None
        updated = []
        for step in range(cfg.inner_steps):
            pred = model(x_hist)
            loss = nmse(pred, y_true)
            if surprise0 is None:
                surprise0 = loss.detach().item()
            active = [g for g in self.groups if self.t % g["period"] == 0]
            params = [p for g in active for p in g["params"]]
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            gi = 0
            for g in active:
                gp = grads[gi: gi + len(g["params"])]
                gi += len(g["params"])
                gn = torch.sqrt(sum((x ** 2).sum() for x in gp if x is not None)).item() \
                    if any(x is not None for x in gp) else 0.0
                if gn == 0.0 or gn > cfg.trust_grad_norm:
                    continue
                scale = min(1.0, cfg.grad_clip / (gn + 1e-12))
                with torch.no_grad():
                    for p, gr in zip(g["params"], gp):
                        if gr is not None:
                            p.add_(gr, alpha=-g["lr"] * scale)
                if step == 0:
                    updated.append(g["name"])

        model.cms.update_anchor(cfg.anchor_decay)
        if cfg.anchor_beta < 1.0:
            model.cms.stabilise_to_anchor(cfg.anchor_beta)
        self._bad = self._bad + 1 if surprise0 > cfg.reset_surprise else 0
        if self._bad >= cfg.reset_patience:
            model.cms.reset_slow_to_anchor()
            if model.titans is not None:
                model.titans.reset_state()
            self._bad = 0

        model.eval()
        rec = {"t": self.t, "surprise": surprise0, "updated": updated}
        self.log.append(rec)
        return rec
