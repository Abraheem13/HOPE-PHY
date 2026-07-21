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
        path = d[f"{split}_path"]
        return LLM4CPDataset(path, p_hist=d.get("p_hist", 16), l_fut=d.get("l_fut", 4),
                             data_key=d.get("data_key"), limit=d.get("limit"))
    raise ValueError(f"Unknown dataset '{name}'")
