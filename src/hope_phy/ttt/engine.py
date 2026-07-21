"""Test-time adaptation (TTA) engine for HOPE-PHY.

Online protocol (zero pilot overhead): at stream step t we predict the next
L instants from the past P. When the stream advances, the *realised* channel
for previously-predicted instants arrives for free from the pilot stream --
these delayed labels drive self-supervised online updates.

Update rule per step t:
  1. surprise_t = per-sample NMSE of the just-labelled prediction.
  2. Titans memory write (every step, cheapest, most plastic).
  3. CMS level k updates only if (t % period_k == 0) AND surprise passes the
     gate; step size lr_k. Slow levels therefore integrate many surprises.
  4. Safeguards: (a) gradient-norm clip; (b) skip-update trust region if
     grad norm explodes; (c) EMA anchor update + stabilisation of the slowest
     level; (d) hard reset-to-anchor if surprise stays catastrophic for
     ``reset_patience`` consecutive steps (pathological-drift recovery).

Ablation switches (for the causal-mechanism table):
  * uniform_periods=True  -> no timescale separation (all levels every step)
  * gate_threshold=0      -> no surprise gating
  * anchor_beta=1.0       -> no EMA stabilisation
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..metrics.nmse import nmse
from ..models.cms.hope_phy import HopePhy


@dataclass
class TTTConfig:
    enabled: bool = True
    lr_slow: float = 1e-5
    lr_ratio: float = 6.0          # lr_k = lr_slow * ratio**k
    gate_threshold: float = 0.0    # update only if surprise > threshold (NMSE)
    grad_clip: float = 1.0
    trust_grad_norm: float = 10.0  # skip update if ||g|| exceeds this
    anchor_decay: float = 0.995
    anchor_beta: float = 0.95      # 1.0 disables stabilisation (ablation)
    reset_patience: int = 50
    reset_surprise: float = 5.0    # NMSE (linear) considered catastrophic
    uniform_periods: bool = False  # ablation: strip timescale separation


class TTTEngine:
    def __init__(self, model: HopePhy, cfg: TTTConfig):
        self.model, self.cfg = model, cfg
        groups = model.cms.param_groups(cfg.lr_slow, cfg.lr_ratio)
        self.levels = [
            {"params": list(g["params"]), "lr": g["lr"],
             "period": 1 if cfg.uniform_periods else g["period"]}
            for g in groups
        ]
        self.t = 0
        self._bad_streak = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------------
    def observe(self, x_hist: torch.Tensor, y_true: torch.Tensor) -> dict:
        """One stream step: label arrives for (x_hist -> y_true); adapt."""
        if not self.cfg.enabled:
            return {"t": self.t, "updated_levels": [], "surprise": None}
        self.t += 1
        model, cfg = self.model, self.cfg

        # --- surprise + delayed-label loss --------------------------------
        model.train()
        pred = model(x_hist)
        loss = nmse(pred, y_true)
        surprise = float(loss)

        # --- Titans write (every step) ------------------------------------
        if model.titans is not None:
            with torch.no_grad():
                feat = model.encode(x_hist)
                tgt_feat = model.encode(y_true) if y_true.shape[1] == x_hist.shape[1] \
                    else feat  # v0: self-key; v1: dedicated target encoder
            model.titans.write(feat, tgt_feat)

        # --- CMS multi-rate updates ---------------------------------------
        updated = []
        if surprise > cfg.gate_threshold:
            grads = torch.autograd.grad(
                loss, [p for lv in self.levels for p in lv["params"]],
                allow_unused=True)
            gi = 0
            for lv in self.levels:
                n = len(lv["params"])
                lv_grads = grads[gi: gi + n]
                gi += n
                if self.t % lv["period"] != 0:
                    continue
                gnorm = torch.sqrt(sum((g ** 2).sum() for g in lv_grads
                                       if g is not None)).item() if any(
                    g is not None for g in lv_grads) else 0.0
                if gnorm > cfg.trust_grad_norm or gnorm == 0.0:
                    continue                                   # trust region
                with torch.no_grad():
                    scale = min(1.0, cfg.grad_clip / (gnorm + 1e-12))
                    for p, g in zip(lv["params"], lv_grads):
                        if g is not None:
                            p.add_(g, alpha=-lv["lr"] * scale)
                updated.append(lv["period"])

        # --- anchor maintenance + catastrophic-drift safeguard ------------
        model.cms.update_anchor(cfg.anchor_decay)
        if cfg.anchor_beta < 1.0:
            model.cms.stabilise_to_anchor(cfg.anchor_beta)
        self._bad_streak = self._bad_streak + 1 if surprise > cfg.reset_surprise else 0
        if self._bad_streak >= cfg.reset_patience:
            model.cms.reset_slow_to_anchor()
            if model.titans is not None:
                model.titans.reset_state()
            self._bad_streak = 0

        rec = {"t": self.t, "surprise": surprise, "updated_levels": updated}
        self.log.append(rec)
        model.eval()
        return rec
