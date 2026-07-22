#!/usr/bin/env python
"""Gradual blended-drift streaming eval (real UMa<->Umi kappa-severity analogue)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data.llm4cp_stream import BlendedStreamCfg, make_blended_stream
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.ttt.engine import TTTConfig, TTTEngine
from hope_phy.utils.seed import get_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--uniform-periods", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--per-speed-limit", type=int, default=30)
    ap.add_argument("--ttt-lr", type=float, default=1e-4)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]

    scfg = BlendedStreamCfg(per_speed_limit=args.per_speed_limit, seed=args.seed)
    probe = next(make_blended_stream(scfg))
    feat_dim = probe[0].shape[-1]

    model = build_model(mcfg["_name"], feat_dim, 4, mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    engine = None
    if args.ttt:
        engine = TTTEngine(model, TTTConfig(
            enabled=True, lr_slow=args.ttt_lr, inner_steps=args.inner_steps,
            uniform_periods=args.uniform_periods,
            anchor_beta=1.0 if args.no_anchor else 0.95,
            freeze_backbone=args.freeze_backbone))

    band_err = defaultdict(list)
    series = []
    for xh, yf, tag in make_blended_stream(scfg):
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            e = float(nmse(model(xh), yf))
        band_err[tag].append(e)
        series.append((tag, e))
        if engine is not None:
            engine.observe(xh, yf)

    band_db = {}
    for tag_b, errs in band_err.items():
        half = errs[len(errs) // 2:]
        band_db[tag_b] = float(10 * np.log10(max(np.mean(half), 1e-12)))

    mode = "ttt" if args.ttt else "frozen"
    for f in ("uniform_periods", "no_anchor", "freeze_backbone"):
        if getattr(args, f):
            mode += "_" + f
    tag = args.tag or f"{Path(args.ckpt).parent.name}_{mode}"
    out = {"tag": tag, "mode": mode, "band_db": band_db, "series": series}
    log = ROOT / "results" / "logs" / f"blended_{tag}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(out, indent=2))

    print(f"\n=== {tag} ===")
    for tag_b in sorted(band_db):
        print(f"  {tag_b} : {band_db[tag_b]:6.2f} dB")


if __name__ == "__main__":
    main()
