#!/usr/bin/env python
"""Hard-cut UMa -> Umi scenario-change evaluation (real drift, no blend)."""
from __future__ import annotations
import argparse, copy, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hope_phy.continual.baselines import BASELINES, BaselineConfig
from hope_phy.data.llm4cp_dataset import LLM4CPDataset
from hope_phy.metrics.nmse import nmse
from hope_phy.models import build_model
from hope_phy.ttt.simple_adapter import SimpleAdapterConfig, SimpleOnlineAdapter
from hope_phy.utils.seed import get_device, seed_everything

UMA_H = "data/llm4cp/Testing Dataset/H_U_his_test.mat"
UMA_P = "data/llm4cp/Testing Dataset/H_U_pre_test.mat"
UMI_H = "data/llm4cp/Testing Dataset/Umi/H_U_his_test.mat"
UMI_P = "data/llm4cp/Testing Dataset/Umi/H_U_pre_test.mat"

def make_stream(per_speed, seed):
    rng = np.random.default_rng(seed)
    uma = LLM4CPDataset(UMA_H, UMA_P, per_speed_limit=per_speed)
    umi = LLM4CPDataset(UMI_H, UMI_P, per_speed_limit=per_speed)
    for ds, tag in [(uma, "uma"), (umi, "umi")]:
        idx = np.arange(len(ds)); rng.shuffle(idx)
        for i in idx:
            xh, yf = ds[int(i)]
            yield xh.unsqueeze(0), yf.unsqueeze(0), tag

def run(state, mcfg, fd, dev, method, lr, steps, seed, per_speed):
    m = build_model(mcfg["_name"], fd, 4, mcfg).to(dev)
    m.load_state_dict(copy.deepcopy(state["model"])); m.eval()
    if method == "frozen":
        ad = None
    elif method == "ours":
        ad = SimpleOnlineAdapter(m, SimpleAdapterConfig(enabled=True, lr=lr, inner_steps=steps, adapt_scope="all"))
    else:
        ad = BASELINES[method](m, BaselineConfig(lr=lr, inner_steps=steps, adapt_scope="all", seed=seed))
    seg = defaultdict(list)
    for xh, yf, tag in make_stream(per_speed, seed):
        xh, yf = xh.to(dev), yf.to(dev)
        with torch.no_grad():
            seg[tag].append(float(nmse(m(xh), yf)))
        if ad is not None:
            ad.observe(xh, yf)
    out = {}
    for t, v in seg.items():
        h = v[len(v)//2:]; mm = np.mean(h)
        out[t] = float("nan") if not np.isfinite(mm) else float(10*np.log10(max(mm,1e-12)))
        q = v[:max(len(v)//4,1)]
        out[t+"_early"] = float(10*np.log10(max(np.mean(q),1e-12)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", type=float, default=6.4e-2)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--per-speed", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_everything(a.seed); dev = get_device()
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mcfg = state["cfg"]["model"]
    fd = next(make_stream(a.per_speed, a.seed))[0].shape[-1]
    methods = ["frozen", "ewc", "replay", "ours"]
    res = {}
    for mth in methods:
        res[mth] = run(state, mcfg, fd, dev, mth, a.lr, a.inner_steps, a.seed, a.per_speed)
        print("[done]", mth)
    log = ROOT/"results"/"logs"/("hardcut_"+Path(a.ckpt).parent.name+"_seed"+str(a.seed)+".json")
    log.parent.mkdir(parents=True, exist_ok=True); log.write_text(json.dumps(res, indent=2))
    fz = res["frozen"]["umi"]
    print("\nmethod    uma    umi   umi_early   gain")
    for mth in methods:
        r = res[mth]
        print("%-8s %6.2f %6.2f  %7.2f   %+.2f" % (mth, r["uma"], r["umi"], r["umi_early"], fz - r["umi"]))

if __name__ == "__main__":
    main()
