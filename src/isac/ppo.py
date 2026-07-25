"""PPO; nested=True partitions params into slow/med/fast groups with geometric
LRs + update periods + EMA anchor on the slow group. Same net, data, and PPO
hyper-params in both arms -- only the update schedule differs."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    obs_dim: int = 10
    act_dim: int = 2
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 8
    minibatch: int = 256
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    nested: bool = False
    lr_slow: float = 5e-5
    lr_med: float = 2e-4
    lr_fast: float = 8e-4
    period_med: int = 2
    period_slow: int = 6
    anchor_decay: float = 0.99
    anchor_beta: float = 0.98


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.l0 = nn.Linear(obs_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.pi_mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(-0.5*torch.ones(act_dim))
        self.v = nn.Linear(hidden, 1)
        self.act_fn = nn.Tanh()

    def forward(self, x):
        h = self.act_fn(self.l0(x))
        h = self.act_fn(self.l1(h))
        h = self.act_fn(self.l2(h))
        return self.pi_mu(h), self.log_std.exp(), self.v(h).squeeze(-1)

    def dist(self, x):
        mu, std, v = self(x)
        return Normal(mu, std), v


class PPO:
    def __init__(self, cfg: PPOConfig, device):
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(device)
        self.update_count = 0
        if not cfg.nested:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
            self.groups = None
        else:
            n = self.net
            slow = list(n.l0.parameters())
            med = list(n.l1.parameters())
            fast = (list(n.l2.parameters()) + list(n.pi_mu.parameters())
                    + [n.log_std] + list(n.v.parameters()))
            self.groups = [
                {"name": "slow", "params": slow, "period": cfg.period_slow},
                {"name": "med", "params": med, "period": cfg.period_med},
                {"name": "fast", "params": fast, "period": 1},
            ]
            self.opt = torch.optim.Adam([
                {"params": slow, "lr": cfg.lr_slow},
                {"params": med, "lr": cfg.lr_med},
                {"params": fast, "lr": cfg.lr_fast},
            ])
            self._anchor = {k: v.detach().clone() for k, v in n.l0.state_dict().items()}

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        d, v = self.net.dist(x)
        a = d.sample()
        return a.cpu().numpy(), d.log_prob(a).sum(-1).item(), v.item()

    def update(self, buf):
        cfg = self.cfg
        obs = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.array(buf["act"]), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(np.array(buf["logp"]), dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(np.array(buf["adv"]), dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(np.array(buf["ret"]), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean())/(adv.std()+1e-8)

        self.update_count += 1
        if self.groups is not None:
            for g in self.groups:
                trainable = (self.update_count % g["period"] == 0)
                for p in g["params"]:
                    p.requires_grad_(trainable)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                mb = idx[s:s+cfg.minibatch]
                d, v = self.net.dist(obs[mb])
                logp = d.log_prob(act[mb]).sum(-1)
                ratio = (logp - logp_old[mb]).exp()
                clip_adv = torch.clamp(ratio, 1-cfg.clip, 1+cfg.clip)*adv[mb]
                pi_loss = -(torch.min(ratio*adv[mb], clip_adv)).mean()
                v_loss = ((v - ret[mb])**2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + cfg.vf_coef*v_loss - cfg.ent_coef*ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.net.parameters() if p.requires_grad], cfg.grad_clip)
                self.opt.step()

        if self.groups is not None:
            for g in self.groups:
                for p in g["params"]:
                    p.requires_grad_(True)
            with torch.no_grad():
                sd = self.net.l0.state_dict()
                for k in self._anchor:
                    self._anchor[k].mul_(cfg.anchor_decay).add_(sd[k], alpha=1-cfg.anchor_decay)
                    sd[k].mul_(cfg.anchor_beta).add_(self._anchor[k], alpha=1-cfg.anchor_beta)
                self.net.l0.load_state_dict(sd)
        return {}


def compute_gae(rews, vals, dones, gamma, lam):
    n = len(rews)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nonterm = 0.0 if dones[t] else 1.0
        next_v = vals[t+1] if t+1 < n else 0.0
        delta = rews[t] + gamma*next_v*nonterm - vals[t]
        adv[t] = last = delta + gamma*lam*nonterm*last
    ret = adv + np.array(vals[:n], dtype=np.float32)
    return adv, ret
