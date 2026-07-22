from __future__ import annotations

from .llm4cp_dataset import LLM4CPDataset
from .synthetic import SyntheticCfg, SyntheticChannelDataset, make_synthetic


def build_dataset(cfg: dict, split: str = "train"):
    name = cfg.get("_name", "synthetic")
    d = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if name == "synthetic":
        d = dict(d)
        d["seed"] = d.get("seed", 0) + {"train": 0, "val": 1000, "test": 2000}[split]
        if split != "train":
            d["n_series"] = max(4, d.get("n_series", 64) // 8)
        return make_synthetic(d)
    if name == "llm4cp":
        # val reuses the train files (LLM4CP has no separate val split);
        # we carve a held-out slice via per_speed_limit for quick validation.
        s = {"train": "train", "val": "train", "test": "test"}[split]
        his = d[f"his_{s}_path"]
        pre = d[f"pre_{s}_path"]
        lim = d.get("val_per_speed_limit") if split == "val" else d.get("per_speed_limit")
        return LLM4CPDataset(his, pre, per_speed_limit=lim)
    raise ValueError(f"Unknown dataset '{name}'")
