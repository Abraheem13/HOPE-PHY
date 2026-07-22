#!/usr/bin/env python
"""Real UMa->Umi streaming eval with delayed-pilot test-time adaptation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_phy.data.llm4cp_stream import LLM4CPStreamCfg, make_llm4cp_stream
from hope_phy.metrics.nmse import StreamingNMSE
from hope_phy.models import build_model
from hope_phy.ttt.engine import TTTConfig, TTTEngine
from hope_phy.utils.seed import get_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--uniform-periods", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--per-speed-limit", type=int, default=30)
    ap.add_argument("--ttt-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = state["cfg"]["model"]

    scfg = LLM4CPStreamCfg(per_speed_limit=args.per_speed_limit, seed=args.seed)
    probe = next(make_llm4cp_stream(scfg))
    feat_dim = probe[0].shape[-1]

    model = build_model(mcfg["_name"], feat_dim, 4, mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    engine = None
    if args.ttt:
        if mcfg["_name"] != "hope_phy":
            raise SystemExit("TTA engine requires a hope_phy checkpoint")
        engine = TTTEngine(model, TTTConfig(
            enabled=True, lr_slow=args.ttt_lr,
            uniform_periods=args.uniform_periods,
            anchor_beta=1.0 if args.no_anchor else 0.95))

    meter = StreamingNMSE()
    for xh, yf, tag in make_llm4cp_stream(scfg):
        xh, yf = xh.to(device), yf.to(device)
        with torch.no_grad():
            pred = model(xh)
        meter.update(pred, yf, tag)
        if engine is not None:
            engine.observe(xh, yf)

    summary = meter.segment_summary()
    mode = ("ttt" if args.ttt else "frozen")
    if args.uniform_periods:
        mode += "_uniform"
    if args.no_anchor:
        mode += "_noanchor"
    tag = args.tag or f"{Path(args.ckpt).parent.name}_{mode}"

    out = {"tag": tag, "mode": mode, "model": mcfg["_name"],
           "summary": summary, "db_series": meter.db_series(),
           "step_tags": meter.tags}
    log = ROOT / "results" / "logs" / f"stream_real_{tag}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(out, indent=2))

    print(f"\n=== {tag} ===")
    for scen, s in summary.items():
        print(f"  [{scen:4s}] steady {s['steady_db']:6.2f} dB | "
              f"degrade {s['degradation_db']:5.2f} dB | "
              f"recover {int(s['recovery_steps']):4d} steps")


if __name__ == "__main__":
    main()
