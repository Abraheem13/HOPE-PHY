#!/usr/bin/env python
from __future__ import annotations
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
BANDS = ["b0.00", "b0.25", "b0.50", "b0.75", "b1.00"]
BETA = [0.0, 0.25, 0.5, 0.75, 1.0]
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.6})
W, H = 3.5, 2.6

def _load(p):
    f = LOGS / p
    return json.loads(f.read_text()) if f.exists() else None

def fig1():
    fr, tt = [], []
    for s in [42, 43, 44, 45, 46]:
        a, b = _load(f"blended_seed_frozen_{s}.json"), _load(f"blended_seed_ttt_{s}.json")
        if a and b:
            fr.append([a["band_db"][x] for x in BANDS]); tt.append([b["band_db"][x] for x in BANDS])
    if not fr:
        print("skip fig1"); return
    fr, tt = np.array(fr), np.array(tt)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.errorbar(BETA, fr.mean(0), yerr=fr.std(0), marker="o", label="Frozen", color="#c0392b")
    ax.errorbar(BETA, tt.mean(0), yerr=tt.std(0), marker="s", label="Adapted", color="#2471a3")
    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Drift severity beta"); ax.set_ylabel("NMSE (dB)"); ax.legend()
    fig.savefig(FIGS / "fig1_crossover.pdf"); plt.close(fig); print("fig1 ok")

def fig2():
    keys = ["frozen", "ewc", "ours", "replay"]
    lab = {"frozen": "Frozen", "ewc": "EWC", "ours": "Online adapt.", "replay": "Replay (ours)"}
    col = {"frozen": "#c0392b", "ewc": "#7d3c98", "ours": "#2471a3", "replay": "#148f77"}
    acc = {k: [] for k in keys}
    for s in [42, 43, 44]:
        d = _load(f"baselines_hope_phy_llm4cp_seed42_seed{s}.json")
        if d:
            for k in keys: acc[k].append([d[k][b] for b in BANDS])
    if not acc["frozen"]:
        print("skip fig2"); return
    fig, ax = plt.subplots(figsize=(W, H))
    for k in keys:
        a = np.array(acc[k])
        ax.errorbar(BETA, a.mean(0), yerr=a.std(0), marker="o", ms=4, label=lab[k], color=col[k])
    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Drift severity beta"); ax.set_ylabel("NMSE (dB)"); ax.legend(fontsize=7)
    fig.savefig(FIGS / "fig2_methods.pdf"); plt.close(fig); print("fig2 ok")

def fig3():
    d = _load("simple_hope_phy_llm4cp_seed42_seed42.json")
    if not d:
        print("skip fig3"); return
    fig, ax = plt.subplots(figsize=(W, H))
    for k, c in zip(["frozen", "head", "head+last", "all"], ["#c0392b", "#e67e22", "#2471a3", "#148f77"]):
        if k in d: ax.plot(BETA, [d[k][b] for b in BANDS], marker="o", ms=4, label=k, color=c)
    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Drift severity beta"); ax.set_ylabel("NMSE (dB)"); ax.legend(fontsize=7)
    fig.savefig(FIGS / "fig3_scope.pdf"); plt.close(fig); print("fig3 ok")

def fig4():
    p = LOGS / "matched_table.csv"
    if not p.exists():
        print("skip fig4"); return
    rows = list(csv.DictReader(p.open()))
    vel = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(figsize=(W, H))
    seen = set()
    for r in rows:
        if r["model"] in seen: continue
        seen.add(r["model"])
        ax.plot(vel, [float(r[f"{v}kmh"]) for v in vel], marker="o", ms=4, label=r["model"])
    ax.set_xlabel("UE speed (km/h)"); ax.set_ylabel("NMSE (dB)"); ax.legend(fontsize=7)
    fig.savefig(FIGS / "fig4_velocity.pdf"); plt.close(fig); print("fig4 ok")

for f in (fig1, fig2, fig3, fig4):
    try: f()
    except Exception as e: print(f"[warn] {f.__name__}: {e}")
print("done ->", FIGS)
