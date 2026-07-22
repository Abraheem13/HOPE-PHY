#!/usr/bin/env python
"""Multi-timescale drift eval: slow scenario ramp + fast severity jitter."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data.llm4cp_stream import MultiTimescaleStreamCfg, make_multitimescale_stream
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.ttt.engine import TTTConfig, TTTEngine
from hope_phy.utils.seed import get_device, seed_everything


def run_variant(state, mcfg, feat_dim, device, scfg, ttt_kwargs, engine_on):
    model = build_model(mcfg["_name"], feat_dim, 4, mcfg).to(device)
    model.load_state_dict(copy.deepcopy(state["model"]))
    model.eval()
    engine = TTTEngine(model, TTTConfig(**ttt_kwargs)) if engine_on else None
    band = defaultdict(list)
    for xh, yf, tag in make_multitimescale_stream(scfg):
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            e = float(nmse(model(xh), yf))
        band[tag].append(e)
        if engine is not None:
            engine.observe(xh, yf)
    return {t: float(10 * np.log10(max(np.mean(v[len(v)//2:]), 1e-12)))
            for t, v in band.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ttt-lr", type=float, default=1e-3)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--jitter-amp", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]
    scfg = MultiTimescaleStreamCfg(seed=args.seed, jitter_amp=args.jitter_amp)
    feat_dim = next(make_multitimescale_stream(scfg))[0].shape[-1]

    base = dict(enabled=True, lr_slow=args.ttt_lr, inner_steps=args.inner_steps)
    variants = {
        "frozen":     (None, False),
        "full":       (dict(base), True),
        "-timescale": (dict(base, uniform_periods=True), True),
        "-anchor":    (dict(base, anchor_beta=1.0), True),
        "-backbone":  (dict(base, freeze_backbone=True), True),
    }

    results = {}
    for name, (kw, on) in variants.items():
        results[name] = run_variant(state, mcfg, feat_dim, device, scfg,
                                     kw or {"enabled": False}, on)
        print(f"[done] {name}")

    log = ROOT / "results" / "logs" / f"multits_{Path(args.ckpt).parent.name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(results, indent=2))

    bands = sorted({b for r in results.values() for b in r})
    hdr = "variant".ljust(14) + " ".join(b.rjust(6) for b in bands)
    print("\n" + hdr)
    for name, r in results.items():
        print(name.ljust(14) + " ".join((f"{r[b]:6.2f}" if b in r else "   nan") for b in bands))


if __name__ == "__main__":
    main()
