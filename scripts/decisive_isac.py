#!/usr/bin/env python
"""DECISIVE PRE-REGISTERED TEST: single vs nested timescale PPO on ISAC.
PASS iff nested mean asymptotic advantage >= 3% AND wins >= 4/5 seeds."""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isac.env import ISACConfig, ISACEnv
from isac.ppo import PPO, PPOConfig, compute_gae


def run_arm(nested, seed, episodes, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ISACEnv(ISACConfig(seed=seed))
    cfg = PPOConfig(obs_dim=env.obs_dim, act_dim=env.act_dim, nested=nested)
    agent = PPO(cfg, device)
    ep_rewards = []
    regime_rho = defaultdict(list)
    for ep in range(episodes):
        obs = env.reset()
        buf = {"obs": [], "act": [], "logp": [], "rew": [], "val": [], "done": []}
        done, total = False, 0.0
        while not done:
            a, logp, v = agent.act(obs)
            nobs, r, done, info = env.step(a)
            buf["obs"].append(obs); buf["act"].append(a)
            buf["logp"].append(logp); buf["rew"].append(r)
            buf["val"].append(v); buf["done"].append(done)
            regime_rho[info["regime"]].append(info["rho"])
            obs = nobs; total += r
        adv, ret = compute_gae(buf["rew"], buf["val"], buf["done"], cfg.gamma, cfg.lam)
        agent.update({"obs": buf["obs"], "act": buf["act"], "logp": buf["logp"],
                      "adv": adv, "ret": ret})
        ep_rewards.append(total / env.cfg.episode_slots)
    n = len(ep_rewards)
    return {"rewards": ep_rewards,
            "asymptotic": float(np.mean(ep_rewards[int(0.75*n):])),
            "early": float(np.mean(ep_rewards[:int(0.4*n)])),
            "regime_rho": {int(k): float(np.mean(v)) for k, v in regime_rho.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    res = {"single": [], "nested": []}
    for seed in range(42, 42 + a.seeds):
        for arm, nested in [("single", False), ("nested", True)]:
            t0 = time.time()
            r = run_arm(nested, seed, a.episodes, device)
            res[arm].append(r)
            print("[seed %d %-7s] asym %.4f early %.4f rho %s (%ds)"
                  % (seed, arm, r["asymptotic"], r["early"],
                     r["regime_rho"], time.time()-t0))
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/logs/decisive_isac.json").write_text(json.dumps(res, indent=2))
    s = np.array([r["asymptotic"] for r in res["single"]])
    nn_ = np.array([r["asymptotic"] for r in res["nested"]])
    se = np.array([r["early"] for r in res["single"]])
    ne = np.array([r["early"] for r in res["nested"]])
    adv = (nn_ - s)/np.abs(s)*100
    wins = int((nn_ > s).sum())
    print("\n================ DECISIVE RESULT ================")
    print("single asym : %.4f +- %.4f" % (s.mean(), s.std()))
    print("nested asym : %.4f +- %.4f" % (nn_.mean(), nn_.std()))
    print("advantage   : %+.1f%%  per-seed: %s" % (adv.mean(), " ".join("%+.1f%%" % x for x in adv)))
    print("early       : single %.4f | nested %.4f" % (se.mean(), ne.mean()))
    print("consistency : nested wins %d/%d" % (wins, len(adv)))
    verdict = "PASS" if (adv.mean() >= 3.0 and wins >= max(4, len(adv)-1)) else "FAIL"
    print("\nPRE-REGISTERED VERDICT:", verdict)


if __name__ == "__main__":
    main()
