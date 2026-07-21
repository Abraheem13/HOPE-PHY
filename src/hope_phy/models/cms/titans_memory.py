"""Titans-style neural long-term memory with surprise-gated online updates.

A small MLP memory M maps keys -> values. At test time it is trained online
by gradient descent on the associative loss ||M(k_t) - v_t||^2, with
  * momentum-accumulated *surprise* (past + momentary), and
  * weight-decay *forgetting*,
following the Titans update
    S_t = eta_t * S_{t-1} - theta_t * grad_t          (surprise w/ momentum)
    M_t = (1 - a_t) * M_{t-1} + S_t                   (update w/ forgetting)
where here (eta, theta, a) are data-independent scalars in v0 and can be made
input-conditioned (self-modifying) in v1. Retrieval is a plain forward pass.

In HOPE-PHY the keys are backbone features of the historical window and the
values are (projected) realised channel targets, so online labels arrive for
free from the pilot stream -- zero-overhead adaptation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TitansMemory(nn.Module):
    def __init__(self, dim: int, mem_hidden: int = 128, lr: float = 1e-2,
                 momentum: float = 0.9, forget: float = 1e-3):
        super().__init__()
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.memory = nn.Sequential(
            nn.Linear(dim, mem_hidden), nn.SiLU(), nn.Linear(mem_hidden, dim)
        )
        self.lr, self.momentum, self.forget = lr, momentum, forget
        self._vel: list[torch.Tensor] | None = None
        self.last_surprise: float = 0.0

    # ----- retrieval (differentiable, used in the main forward pass) -------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.memory(self.key_proj(x))

    # ----- online write (called by the TTT engine, outside autograd graph) -
    @torch.enable_grad()
    def write(self, feat: torch.Tensor, target_feat: torch.Tensor) -> float:
        """One surprise-gated memory step; returns the associative loss."""
        k = self.key_proj(feat.detach())
        v = self.val_proj(target_feat.detach())
        pred = self.memory(k)
        loss = F.mse_loss(pred, v)
        grads = torch.autograd.grad(loss, list(self.memory.parameters()))
        with torch.no_grad():
            if self._vel is None:
                self._vel = [torch.zeros_like(g) for g in grads]
            for p, g, vel in zip(self.memory.parameters(), grads, self._vel):
                vel.mul_(self.momentum).add_(g, alpha=-self.lr)   # surprise momentum
                p.mul_(1.0 - self.forget).add_(vel)               # forgetting + write
        self.last_surprise = float(loss)
        return self.last_surprise

    @torch.no_grad()
    def reset_state(self) -> None:
        self._vel = None
        self.last_surprise = 0.0
