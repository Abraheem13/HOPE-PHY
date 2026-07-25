#!/usr/bin/env python
"""Long-horizon probe: train-free eval OR full pipeline for P=8->L=8.

  train:  python scripts/probe_long.py train  [--epochs 150]
  eval :  python scripts/probe_long.py eval   [--lr 6.4e-2 --inner-steps 8]

Pre-registered pass criteria (decided before running):
  PASS iff frozen Umi is >=3 dB worse than frozen UMa AND the best adapter
  recovers >=1.5 dB of that gap. Otherwise Idea 1 is closed.
"""
from __future__ import annotations
import argparse, copy, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hope_phy.continual.baselines import BASELINES, BaselineConfig
from hope_phy.data.long_horizon import LongHorizonDataset
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.train.trainer import Trainer
from hope_phy.ttt.simple_adapter import SimpleAdapterConfig, SimpleOnlineAdapter
from hope_phy.utils.seed import get_device, seed_everything

TRAIN_H = "data/llm4cp/Training Dataset/H_U_his_train.mat"
UMA_H   = "data/llm4cp/Testing Dataset/H_U_his_test.mat"
UMI_H   = "data/llm4cp/Testing Dataset/Umi/H_U_his_test.mat"
CKPT    = "results/checkpoints/longhorizon_transformer_seed42"

MCFG = {"_name": "transformer", "d_model": 512, "nhead": 8, "layers": 4,
        "ff": 1024, "dropout": 0.1}


def do_train(a):
    seed_everything(42)
    full = LongHorizonDataset(TRAIN_H)
    n_val = 200
    class Slice(torch.utils.data.Dataset):
        def __init__(s, ds, lo, hi): s.ds, s.lo, s.hi = ds, lo, hi
        def __len__(s): return s.hi - s.lo
        def __getitem__(s, i): return s.ds[s.lo + i]
    tr, va = Slice(full, 0, len(full)-n_val), Slice(full, len(full)-n_val, len(full))
    model = build_model("transformer", full.feat_dim, 8, MCFG)
    cfg = {"train": {"epochs": a.epochs, "batch_size": 256, "lr": 1e-3,
                     "lr_step": 60, "lr_gamma": 0.3, "grad_clip": 1.0, "workers": 2},
           "model": MCFG, "data": {"_name": "long_horizon"}}
    Trainer(model, cfg, CKPT, get_device()).fit(tr, va)


def stream(seed, per_speed):
    rng = np.random.default_rng(seed)
    for path, tag in [(UMA_H, "uma"), (UMI_H, "umi")]:
        ds = LongHorizonDataset(path, per_speed_limit=per_speed)
        idx = np.arange(len(ds)); rng.shuffle(idx)
        for i in idx:
            xh, yf = ds[int(i)]
            yield xh.unsqueeze(0), yf.unsqueeze(0), tag


def do_eval(a):
    seed_everything(a.seed)
    dev = get_device()
    state = torch.load(Path(CKPT) / "best.pt", map_location=dev, weights_only=False)
    fd = state["model"]["inp.weight"].shape[1]
    methods = ["frozen", "replay", "ours"]
    res = {}
    for mth in methods:
        m = build_model("transformer", fd, 8, MCFG).to(dev)
        m.load_state_dict(copy.deepcopy(state["model"])); m.eval()
        if mth == "frozen":
            ad = None
        elif mth == "ours":
            ad = SimpleOnlineAdapter(m, SimpleAdapterConfig(enabled=True, lr=a.lr,
                 inner_steps=a.inner_steps, adapt_scope="all"))
        else:
            ad = BASELINES[mth](m, BaselineConfig(lr=a.lr, inner_steps=a.inner_steps,
                 adapt_scope="all", seed=a.seed))
        seg = defaultdict(list)
        for xh, yf, tag in stream(a.seed, a.per_speed):
            xh, yf = xh.to(dev), yf.to(dev)
            with torch.no_grad():
                seg[tag].append(float(nmse(m(xh), yf)))
            if ad is not None:
                ad.observe(xh, yf)
        res[mth] = {t: float(10*np.log10(max(np.mean(v[len(v)//2:]), 1e-12)))
                    for t, v in seg.items()}
        print("[done]", mth, res[mth])
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/logs/probe_long.json").write_text(json.dumps(res, indent=2))
    fz = res["frozen"]
    drop = fz["umi"] - fz["uma"]
    print("\nfrozen UMa %.2f | frozen Umi %.2f | degradation %.2f dB" % (fz["uma"], fz["umi"], drop))
    best = min(res[m]["umi"] for m in methods if m != "frozen")
    rec = fz["umi"] - best
    print("best adapted Umi %.2f | recovery %.2f dB" % (best, rec))
    verdict = "PASS" if (drop >= 3.0 and rec >= 1.5) else "FAIL"
    print("\nPRE-REGISTERED VERDICT:", verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=6.4e-2)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--per-speed", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    (do_train if a.mode == "train" else do_eval)(a)

if __name__ == "__main__":
    main()
