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
        if split == "test":
            return LLM4CPDataset(d["his_test_path"], d["pre_test_path"])
        full = LLM4CPDataset(d["his_train_path"], d["pre_train_path"])
        val_blocks = d.get("val_blocks", 20)
        ns = full.n_speeds
        n_val = val_blocks * ns
        if split == "val":
            full.samples = full.samples[-n_val:]
            full.speed_index = full.speed_index[-n_val:]
        else:
            full.samples = full.samples[:-n_val]
            full.speed_index = full.speed_index[:-n_val]
        return full
    raise ValueError(f"Unknown dataset '{name}'")
