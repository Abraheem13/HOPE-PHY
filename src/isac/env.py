"""ISAC multi-timescale control environment (fast fading / medium target / slow regime)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ISACConfig:
    n_antennas: int = 32
    slot_len: float = 1e-3
    episode_slots: int = 400
    init_range: tuple = (60.0, 120.0)
    init_speed: tuple = (5.0, 20.0)
    dyn_dt: float = 0.05
    vel_revert: float = 0.995
    # regimes: (clutter meas-noise std [m], manoeuvre accel std, rician K)
    regimes: tuple = ((0.5, 0.3, 10.0),   # calm : near-static -> sense rarely (best k=32)
                      (2.0, 40.0, 4.0),   # agile: violent manoeuvre -> sense often (best k=8)
                      (1.0, 8.0, 7.0))    # mixed (best k=16)
    regime_period: int = 100
    fading_coherence: int = 1
    w_rate: float = 1.0
    w_track: float = 1.0
    w_energy: float = 0.35
    d_scale: float = 5.0
    snr0_comm: float = 8.0
    snr0_sense: float = 50.0
    beamwidth: float = 0.15
    seed: int = 0


class ISACEnv:
    def __init__(self, cfg: ISACConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.obs_dim = 10
        self.act_dim = 2

    def reset(self):
        c = self.cfg
        r0 = self.rng.uniform(*c.init_range)
        ang = self.rng.uniform(-np.pi/3, np.pi/3)
        sp = self.rng.uniform(*c.init_speed)
        head = self.rng.uniform(0, 2*np.pi)
        self.x = np.array([r0*np.cos(ang), r0*np.sin(ang), sp*np.cos(head), sp*np.sin(head)])
        self.mu = self.x + self.rng.normal(0, 2.0, 4)
        self.P = np.diag([4.0, 4.0, 4.0, 4.0])
        self.t = 0
        self.regime_idx = int(self.rng.integers(len(c.regimes)))
        self._new_fading()
        return self._obs()

    def _regime(self):
        return self.cfg.regimes[self.regime_idx]

    def _new_fading(self):
        k = self._regime()[2]
        los = np.sqrt(k/(k+1))
        nlos = self.rng.normal(0, np.sqrt(1/(2*(k+1))), 2)
        self.g = (los + nlos[0])**2 + nlos[1]**2

    def _true_angle(self):
        return np.arctan2(self.x[1], self.x[0])

    def _pred_angle(self):
        return np.arctan2(self.mu[1], self.mu[0])

    def _obs(self):
        c = self.cfg
        rng_est = np.linalg.norm(self.mu[:2])
        return np.array([
            self.mu[0]/100, self.mu[1]/100, self.mu[2]/20, self.mu[3]/20,
            np.sqrt(self.P[0,0]+self.P[1,1])/10,
            np.sqrt(self.P[2,2]+self.P[3,3])/10,
            self.g, rng_est/100,
            np.sin(2*np.pi*self.t/c.regime_period),
            np.cos(2*np.pi*self.t/c.regime_period),
        ], dtype=np.float32)

    def step(self, action):
        c = self.cfg
        clutter_std, accel_std, _k = self._regime()
        rho = float(np.clip(action[0], 0.0, 1.0))
        beam_off = float(np.clip(action[1], -1.0, 1.0)) * 0.3

        dt = c.dyn_dt
        rv = c.vel_revert
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,rv,0],[0,0,0,rv]])
        acc = self.rng.normal(0, accel_std, 2)
        self.x = F @ self.x + np.array([0.5*dt*dt*acc[0], 0.5*dt*dt*acc[1], dt*acc[0], dt*acc[1]])
        q = (accel_std*dt)**2
        Q = q*np.array([[.25*dt*dt,0,.5*dt,0],[0,.25*dt*dt,0,.5*dt],[.5*dt,0,1,0],[0,.5*dt,0,1]])
        self.mu = F @ self.mu
        self.P = F @ self.P @ F.T + Q

        beam_ang = self._pred_angle() + beam_off
        point_err = abs(np.arctan2(np.sin(beam_ang - self._true_angle()),
                                   np.cos(beam_ang - self._true_angle())))
        point_gain = np.exp(-(point_err/c.beamwidth)**2)
        rng_true = np.linalg.norm(self.x[:2])
        sense_snr = c.snr0_sense * rho * point_gain / max((rng_true/100)**2, 1e-3)
        if rho > 0.02 and sense_snr > 0.5:
            meas_std = clutter_std / np.sqrt(max(sense_snr, 1e-6))
            z = self.x[:2] + self.rng.normal(0, meas_std, 2)
            H = np.array([[1,0,0,0],[0,1,0,0]])
            R = (meas_std**2)*np.eye(2)
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.mu = self.mu + K @ (z - H @ self.mu)
            self.P = (np.eye(4) - K @ H) @ self.P

        comm_snr = c.snr0_comm * (1.0 - rho) * self.g
        rate = np.log2(1.0 + comm_snr)
        track_rmse = np.linalg.norm(self.x[:2] - self.mu[:2])
        r = c.w_rate*rate + c.w_track*np.exp(-track_rmse/c.d_scale) - c.w_energy*rho

        self.t += 1
        if self.t % c.fading_coherence == 0:
            self._new_fading()
        if self.t % c.regime_period == 0:
            self.regime_idx = int(self.rng.integers(len(c.regimes)))

        done = self.t >= c.episode_slots
        info = {"rate": rate, "track_rmse": track_rmse, "rho": rho,
                "regime": self.regime_idx, "point_gain": point_gain}
        return self._obs(), float(r), done, info
