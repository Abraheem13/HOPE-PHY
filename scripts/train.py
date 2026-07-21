#!/usr/bin/env python
"""Train any predictor.

Examples:
    python scripts/train.py model=lstm data=synthetic
    python scripts/train.py model=hope_phy data=llm4cp train.epochs=500 train.batch_size=512
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data import build_dataset            # noqa: E402
from hope_phy.models import build_model            # noqa: E402
from hope_phy.train.trainer import Trainer         # noqa: E402
from hope_phy.utils.config import load_config, save_config  # noqa: E402
from hope_phy.utils.seed import get_device, seed_everything  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs", sys.argv[1:])
    seed_everything(cfg["seed"])
    device = get_device()

    train_ds = build_dataset(cfg["data"], "train")
    val_ds = build_dataset(cfg["data"], "val")
    feat_dim = train_ds.feat_dim
    l_fut = cfg["data"].get("l_fut", 4)

    model = build_model(cfg["model"]["_name"], feat_dim, l_fut, cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    run = f"{cfg['model']['_name']}_{cfg['data']['_name']}_seed{cfg['seed']}"
    out = ROOT / cfg["out_dir"] / "checkpoints" / run
    print(f"model={cfg['model']['_name']} params={n_params:,} device={device} "
          f"train={len(train_ds)} val={len(val_ds)} feat_dim={feat_dim} -> {out}")

    save_config(cfg, out / "config.yaml")
    result = Trainer(model, cfg, out, device).fit(train_ds, val_ds)
    print("RESULT:", result)


if __name__ == "__main__":
    main()
