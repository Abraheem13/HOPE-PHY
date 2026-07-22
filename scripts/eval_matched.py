#!/usr/bin/env python
"""Per-velocity matched-protocol evaluation on the LLM4CP test set."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data.llm4cp_dataset import LLM4CPDataset
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.utils.seed import get_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--his", default="data/llm4cp/Testing Dataset/H_U_his_test.mat")
    ap.add_argument("--pre", default="data/llm4cp/Testing Dataset/H_U_pre_test.mat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]

    ds = LLM4CPDataset(args.his, args.pre)
    model = build_model(mcfg["_name"], ds.feat_dim, 4, mcfg).to(device).eval()
    model.load_state_dict(state["model"])

    dl = DataLoader(ds, batch_size=256, shuffle=False)
    preds, tgts = [], []
    with torch.no_grad():
        for xh, yf in dl:
            preds.append(model(xh.to(device)).cpu())
            tgts.append(yf)
    preds, tgts = torch.cat(preds), torch.cat(tgts)

    per_sample = nmse(preds, tgts, reduce=False).numpy()
    speed_idx = ds.speed_index
    speeds = sorted(set(speed_idx.tolist()))
    kmh = [(s + 1) * 10 for s in speeds]
    per_speed_db = {}
    for s, v in zip(speeds, kmh):
        vals = per_sample[speed_idx == s]
        per_speed_db[v] = float(10 * np.log10(max(vals.mean(), 1e-12)))
    overall_db = float(10 * np.log10(max(per_sample.mean(), 1e-12)))

    run = Path(args.ckpt).parent.name
    out = {"run": run, "model": mcfg["_name"], "overall_nmse_db": overall_db,
           "per_speed_db": per_speed_db}
    log = ROOT / "results" / "logs" / f"matched_{run}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(out, indent=2))

    csv_path = ROOT / "results" / "logs" / "matched_table.csv"
    header = ["run", "model", "overall_db"] + [f"{v}kmh" for v in kmh]
    row = [run, mcfg["_name"], f"{overall_db:.3f}"] + \
          [f"{per_speed_db[v]:.3f}" for v in kmh]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"\n=== {mcfg['_name']} | overall {overall_db:.2f} dB ===")
    for v in kmh:
        print(f"  {v:3d} km/h : {per_speed_db[v]:6.2f} dB")


if __name__ == "__main__":
    main()
