#!/usr/bin/env python
"""Fading-opportunistic index policy vs fixed duty-cycle vs always/never.

Index: sense when   trace(P_pos) / (eps + rate_forgone(g))  >  tau
  -> sense when tracking uncertainty is high AND/OR the comm channel is in a
     deep fade (cheap to give up). This exploits the ISAC-specific fact that
     the OPPORTUNITY COST of sensing fluctuates with fading -- ignored by
     fixed duty-cycle schedules and not discovered by PPO.
"""
import sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isac.env import ISACEnv, ISACConfig


def run(policy, seed=0, episodes=20):
    tot = []
    for ep in range(episodes):
        env = ISACEnv(ISACConfig(seed=seed*1000+ep))
        obs = env.reset(); done = False; s = 0.0
        while not done:
            a = policy(env)
            obs, r, done, info = env.step(a)
            s += r
        tot.append(s / env.cfg.episode_slots)
    return float(np.mean(tot)), float(np.std(tot))


def duty(k, rho=0.6):
    def p(env):
        return np.array([rho if env.t % k == 0 else 0.0, 0.0])
    return p


def index_policy(tau, rho=0.6, eps=0.5):
    def p(env):
        unc = np.sqrt(env.P[0, 0] + env.P[1, 1])          # position uncertainty
        cost = np.log2(1.0 + env.cfg.snr0_comm * env.g)   # rate we would forgo
        idx = unc / (eps + cost)
        return np.array([rho if idx > tau else 0.0, 0.0])
    return p


if __name__ == "__main__":
    print("baselines:")
    for k in [4, 8, 16, 32, 64]:
        m, s = run(duty(k))
        print("  duty k=%-3d  %.4f +- %.4f" % (k, m, s))
    print("\nfading-opportunistic index policy:")
    best = (None, -9e9)
    for tau in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
        m, s = run(index_policy(tau))
        flag = ""
        if m > best[1]:
            best = (tau, m); flag = "  <-- best"
        print("  tau=%-4.1f    %.4f +- %.4f%s" % (tau, m, s, flag))
    print("\nbest index tau=%.1f -> %.4f" % best)
