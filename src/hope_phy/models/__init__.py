from __future__ import annotations

from torch import nn

from .baselines.rnn import RecurrentPredictor
from .baselines.transformer import TransformerPredictor
from .cms.hope_phy import HopePhy


def build_model(name: str, feat_dim: int, l_fut: int, cfg: dict) -> nn.Module:
    m = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if name in ("rnn", "lstm", "gru"):
        return RecurrentPredictor(feat_dim, l_fut, cell=name, **m)
    if name == "transformer":
        return TransformerPredictor(feat_dim, l_fut, **m)
    if name == "hope_phy":
        return HopePhy(feat_dim, l_fut, **m)
    raise ValueError(f"Unknown model '{name}'")
