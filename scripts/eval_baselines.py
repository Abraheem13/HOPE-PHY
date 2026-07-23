#!/usr/bin/env python
"""Head-to-head: our adapter vs continual-learning baselines under drift."""
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

from hope_phy.continual.baselines import BASELINES, BaselineConfig
from hope_phy.data.llm4cp_stream import BlendedStreamCfg, make_blended_stream
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.ttt.simple_adapter import SimpleAdapterConfig, SimpleOnlineAdapter
from hope_phy.utils.seed import get_device, seed_everything


def run(state, mcfg, feat_dim, device, scfg, method, lr, steps, seed):
    model = build_model(mcfg["_name"], feat_dim, 4, mcfg).to(device)
    model.load_state_dict(copy.deepcopy(state["model"]))
    model.eval()

    if method == "frozen":
        adapter = None
    elif method == "ours":
        adapter = SimpleOnlineAdapter(model, SimpleAdapterConfig(
            enabled=True, lr=lr, inner_steps=steps, adapt_scope="all"))
    else:
        cfg = BaselineConfig(lr=lr, inner_steps=steps, adapt_scope="all", seed=seed)
        adapter = BASELINES[method](model, cfg)

    band = defaultdict(list)
    for xh, yf, tag in make_blended_stream(scfg):
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            band[tag].append(float(nmse(model(xh), yf)))
        if adapter is not None:
            adapter.observe(xh, yf)
    out = {}
    for t, v in band.items():
        m = np.mean(v[len(v)//2:])
        out[t] = float("nan") if not np.isfinite(m) else float(10*np.log10(max(m,1e-12)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", type=float, default=6.4e-2)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]
    scfg = BlendedStreamCfg(seed=args.seed)
    feat_dim = next(make_blended_stream(scfg))[0].shape[-1]

    methods = ["frozen", "naive", "ewc", "replay", "ours"]
    results = {}
    for m in methods:
        results[m] = run(state, mcfg, feat_dim, device, scfg, m,
                         args.lr, args.inner_steps, args.seed)
        print(f"[done] {m}")

    log = ROOT / "results" / "logs" / f"baselines_{Path(args.ckpt).parent.name}_seed{args.seed}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(results, indent=2))

    bands = sorted({b for r in results.values() for b in r})
    print("\n" + "method".ljust(10) + " ".join(b.rjust(7) for b in bands))
    for m in methods:
        r = results[m]
        print(m.ljust(10) + " ".join((f"{r[b]:7.2f}" if b in r else "    nan") for b in bands))


if __name__ == "__main__":
    main()
