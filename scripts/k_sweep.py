#!/usr/bin/env python
"""Value of fading-opportunistic sensing vs fading severity (Rician K sweep).

Premise: the opportunity cost of radar sensing is the forgone communication
rate, which varies with fading. Under near-LOS (high K) there is little
variation to exploit; under NLOS/Rayleigh (K->0) deep fades are frequent and
sensing during them should be nearly free. We therefore sweep K and measure
  gain(K) = best index policy - best fixed duty-cycle
This yields a physical relationship either way.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isac.env import ISACEnv, ISACConfig


def make_cfg(K, seed):
    base = ISACConfig()
    regs = tuple((c, a, K) for (c, a, _k) in base.regimes)   # override K only
    return ISACConfig(seed=seed, regimes=regs)


def run(policy, K, seed=0, episodes=25):
    tot = []
    for ep in range(episodes):
        env = ISACEnv(make_cfg(K, seed * 1000 + ep))
        env.reset(); done = False; s = 0.0
        while not done:
            _, r, done, _ = env.step(policy(env))
            s += r
        tot.append(s / env.cfg.episode_slots)
    return float(np.mean(tot)), float(np.std(tot) / np.sqrt(len(tot)))


def duty(k, rho=0.6):
    return lambda env: np.array([rho if env.t % k == 0 else 0.0, 0.0])


def index_policy(tau, rho=0.6, eps=0.5):
    def p(env):
        unc = np.sqrt(env.P[0, 0] + env.P[1, 1])
        cost = np.log2(1.0 + env.cfg.snr0_comm * env.g)
        return np.array([rho if unc / (eps + cost) > tau else 0.0, 0.0])
    return p


if __name__ == "__main__":
    Ks = [0.0, 0.5, 1.0, 2.0, 4.0, 10.0]      # Rayleigh -> strong LOS
    print(" K     best-duty        best-index       gain")
    for K in Ks:
        d = max((run(duty(k), K) for k in [4, 8, 16, 32]), key=lambda x: x[0])
        i = max((run(index_policy(t), K) for t in [0.3, 0.5, 0.7, 1.0, 1.5]),
                key=lambda x: x[0])
        print("%5.1f  %.4f+-%.4f  %.4f+-%.4f  %+.4f"
              % (K, d[0], d[1], i[0], i[1], i[0] - d[0]))
    print("\nInterpretation: gain should GROW as K->0 (deep fades frequent).")
    print("Flat/zero gain across K => fading-opportunism has no value here.")
