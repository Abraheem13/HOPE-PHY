"""Offline (pre-deployment) trainer.

Works for every model in the repo. For HOPE-PHY it builds a multi-rate Adam
via ``model.param_groups`` (geometric LR spectrum across CMS levels) --
training-time nesting; test-time nesting is handled by ttt.engine.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..metrics.nmse import nmse, nmse_db


class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, out_dir: str | Path,
                 device: torch.device):
        self.model = model.to(device)
        self.cfg, self.device = cfg, device
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        tr = cfg["train"]
        if hasattr(model, "param_groups"):
            groups = model.param_groups(tr["lr"], tr["cms_lr_slow"], tr["cms_lr_ratio"])
            self.opt = torch.optim.Adam(
                [{"params": g["params"], "lr": g["lr"]} for g in groups])
        else:
            self.opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=tr.get("lr_step", 150), gamma=tr.get("lr_gamma", 0.1))
        self.grad_clip = tr.get("grad_clip", 1.0)
        self.history: list[dict] = []

    def fit(self, train_ds, val_ds) -> dict:
        tr = self.cfg["train"]
        dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                        num_workers=tr.get("workers", 2), drop_last=True)
        vl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False)
        best = float("inf")
        for epoch in range(tr["epochs"]):
            t0 = time.time()
            self.model.train()
            tot, nb = 0.0, 0
            for xh, yf in dl:
                xh, yf = xh.to(self.device), yf.to(self.device)
                loss = nmse(self.model(xh), yf)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                # EMA anchor tracks the slow level during offline training too.
                if hasattr(self.model, "cms"):
                    self.model.cms.update_anchor(tr.get("anchor_decay", 0.995))
                tot += float(loss); nb += 1
            self.sched.step()
            val_db = self.evaluate(vl)
            rec = {"epoch": epoch, "train_nmse": tot / max(nb, 1),
                   "val_nmse_db": val_db, "sec": time.time() - t0}
            self.history.append(rec)
            print(f"[{epoch:03d}] train NMSE {rec['train_nmse']:.4f} | "
                  f"val {val_db:6.2f} dB | {rec['sec']:.1f}s")
            if val_db < best:
                best = val_db
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg},
                           self.out / "best.pt")
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2))
        return {"best_val_nmse_db": best}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, tgts = [], []
        for xh, yf in loader:
            preds.append(self.model(xh.to(self.device)).cpu())
            tgts.append(yf)
        return nmse_db(torch.cat(preds), torch.cat(tgts))
