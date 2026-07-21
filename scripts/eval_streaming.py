#!/usr/bin/env python
"""Streaming scenario-transition evaluation.

    python scripts/eval_streaming.py --ckpt results/checkpoints/<run>/best.pt \
        [--ttt] [--uniform-periods] [--no-anchor]

Prediction happens BEFORE the label is revealed to the TTT engine at each
step, so adaptation never leaks into the reported metric.
Outputs JSON: per-segment steady NMSE (dB), degradation depth, recovery time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data.streams import StreamCfg, make_stream        # noqa: E402
from hope_phy.metrics.nmse import StreamingNMSE                 # noqa: E402
from hope_phy.models import build_model                          # noqa: E402
from hope_phy.ttt.engine import TTTConfig, TTTEngine             # noqa: E402
from hope_phy.utils.seed import get_device, seed_everything      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--uniform-periods", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]
    dcfg = state["cfg"]["data"]
    feat_dim = 2 * dcfg.get("n_rb", 12) * dcfg.get("n_ant", 4)
    model = build_model(mcfg["_name"], feat_dim, dcfg.get("l_fut", 4), mcfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    engine = None
    if args.ttt:
        engine = TTTEngine(model, TTTConfig(
            uniform_periods=args.uniform_periods,
            anchor_beta=1.0 if args.no_anchor else 0.95))

    meter = StreamingNMSE()
    stream = make_stream(StreamCfg(seed=args.seed,
                                   n_rb=dcfg.get("n_rb", 12),
                                   n_ant=dcfg.get("n_ant", 4)))
    for xh, yf, tag in stream:
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            pred = model(xh)                 # predict BEFORE label reveal
        meter.update(pred, yf, tag)
        if engine is not None:
            engine.observe(xh, yf)           # then adapt on revealed label

    summary = meter.segment_summary()
    mode = "ttt" if args.ttt else "frozen"
    out = ROOT / "results" / "logs" / f"stream_{Path(args.ckpt).parent.name}_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": mode, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
