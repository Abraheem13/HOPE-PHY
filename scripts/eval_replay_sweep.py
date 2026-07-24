#!/usr/bin/env python
"""Replay buffer-size sweep + multi-seed aggregation."""
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

from hope_phy.continual.baselines import BaselineConfig, ReplayAdapter
from hope_phy.data.llm4cp_stream import BlendedStreamCfg, make_blended_stream
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.utils.seed import get_device, seed_everything

BANDS = ["b0.00", "b0.25", "b0.50", "b0.75", "b1.00"]


def run(state, mcfg, feat_dim, device, scfg, buf, lr, steps, seed):
    model = build_model(mcfg["_name"], feat_dim, 4, mcfg).to(device)
    model.load_state_dict(copy.deepcopy(state["model"]))
    model.eval()
    adapter = None
    if buf is not None:
        adapter = ReplayAdapter(model, BaselineConfig(
            lr=lr, inner_steps=steps, adapt_scope="all",
            buffer_size=buf, replay_batch=min(8, buf), seed=seed))
    band = defaultdict(list)
    for xh, yf, tag in make_blended_stream(scfg):
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            band[tag].append(float(nmse(model(xh), yf)))
        if adapter is not None:
            adapter.observe(xh, yf)
    out = {}
    for t, v in band.items():
        m = np.mean(v[len(v) // 2:])
        out[t] = float("nan") if not np.isfinite(m) else float(10 * np.log10(max(m, 1e-12)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", type=float, default=6.4e-2)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--buffers", type=int, nargs="+", default=[32, 128, 512])
    args = ap.parse_args()

    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]
    acc = defaultdict(lambda: defaultdict(list))

    for seed in args.seeds:
        seed_everything(seed)
        scfg = BlendedStreamCfg(seed=seed)
        feat_dim = next(make_blended_stream(scfg))[0].shape[-1]
        r = run(state, mcfg, feat_dim, device, scfg, None, args.lr, args.inner_steps, seed)
        for b in BANDS:
            acc["frozen"][b].append(r[b])
        for buf in args.buffers:
            r = run(state, mcfg, feat_dim, device, scfg, buf,
                    args.lr, args.inner_steps, seed)
            for b in BANDS:
                acc[f"buf{buf}"][b].append(r[b])
        print(f"[seed {seed} done]")

    res = {k: {b: [float(np.mean(v[b])), float(np.std(v[b]))] for b in BANDS}
           for k, v in acc.items()}
    log = ROOT / "results" / "logs" / "replay_sweep.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(res, indent=2))

    print("\n" + "method".ljust(10) + " ".join(b.rjust(13) for b in BANDS))
    for k in res:
        print(k.ljust(10) + " ".join(f"{res[k][b][0]:6.2f}±{res[k][b][1]:4.2f}" for b in BANDS))
    fz = res["frozen"]["b1.00"][0]
    print("\ngain over frozen at max drift:")
    for k in res:
        if k != "frozen":
            print(f"  {k:8s} {fz - res[k]['b1.00'][0]:+.2f} dB")


if __name__ == "__main__":
    main()
