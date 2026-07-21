"""Continuum Memory System (CMS) for channel prediction.

A chain of K MemoryBlocks with geometrically spaced update periods
    period_k = base_period ** (K - 1 - k),  k = 0 (slow) ... K-1 (fast)
and geometrically spaced training learning rates. The slowest level carries
an EMA anchor (our OJCOMS continuum-memory signature) that resists overwrite
during test-time adaptation and enables safe reset after pathological drift.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .memory_block import MemoryBlock


class ContinuumMemory(nn.Module):
    def __init__(self, dim: int, n_levels: int = 3, base_period: int = 4,
                 hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_levels = n_levels
        periods = [base_period ** (n_levels - 1 - k) for k in range(n_levels)]
        self.blocks = nn.ModuleList(
            MemoryBlock(dim, level=k, period=periods[k], hidden_mult=hidden_mult,
                        dropout=dropout)
            for k in range(n_levels)
        )
        # EMA anchor of the slowest level's parameters (registered lazily).
        self._anchor: dict[str, torch.Tensor] | None = None

    # ----- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:                       # slow -> fast chain
            x = blk(x)
        return x

    # ----- parameter groups for multi-rate optimisation --------------------
    def param_groups(self, lr_slow: float, lr_ratio: float) -> list[dict]:
        """Geometric LR spectrum: level k gets lr_slow * lr_ratio**k."""
        return [
            {"params": list(blk.parameters()),
             "lr": lr_slow * (lr_ratio ** blk.level),
             "level": blk.level, "period": blk.period}
            for blk in self.blocks
        ]

    # ----- EMA anchor on slowest level -------------------------------------
    @torch.no_grad()
    def update_anchor(self, decay: float = 0.995) -> None:
        slow = self.blocks[0]
        if self._anchor is None:
            self._anchor = {n: p.detach().clone() for n, p in slow.named_parameters()}
            return
        for n, p in slow.named_parameters():
            self._anchor[n].mul_(decay).add_(p.detach(), alpha=1 - decay)

    @torch.no_grad()
    def stabilise_to_anchor(self, beta: float = 0.95) -> None:
        """theta_slow <- beta * theta_slow + (1-beta) * anchor  (OJCOMS Eq. 5 analogue)."""
        if self._anchor is None:
            return
        for n, p in self.blocks[0].named_parameters():
            p.mul_(beta).add_(self._anchor[n], alpha=1 - beta)

    @torch.no_grad()
    def reset_slow_to_anchor(self) -> None:
        """Hard safeguard: restore the slow level from the anchor."""
        if self._anchor is None:
            return
        for n, p in self.blocks[0].named_parameters():
            p.copy_(self._anchor[n])

    def clone_anchor_state(self) -> dict | None:
        return copy.deepcopy(self._anchor)
