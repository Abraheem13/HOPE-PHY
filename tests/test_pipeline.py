"""Smoke + correctness tests. Run: pytest tests/ -q"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hope_phy.data.synthetic import SyntheticCfg, SyntheticChannelDataset
from hope_phy.metrics.nmse import nmse, nmse_db
from hope_phy.models import build_model
from hope_phy.models.cms.hope_phy import HopePhy
from hope_phy.ttt.engine import TTTConfig, TTTEngine

B, P, L, D = 8, 16, 4, 96  # 12 RB * 4 ant * 2 (Re/Im)


def test_synthetic_dataset_shapes():
    ds = SyntheticChannelDataset(SyntheticCfg(n_series=2, t_steps=60))
    xh, yf = ds[0]
    assert xh.shape == (16, 96) and yf.shape == (4, 96)
    assert torch.isfinite(xh).all() and torch.isfinite(yf).all()


def test_nmse_perfect_and_scale():
    y = torch.randn(B, L, D)
    assert nmse(y, y) < 1e-10
    assert nmse_db(0.9 * y, y) < 0  # 10log10(0.01) = -20 dB


def test_all_models_forward():
    x = torch.randn(B, P, D)
    for name, cfg in [("lstm", {"hidden": 32, "layers": 1}),
                      ("transformer", {"d_model": 32, "nhead": 4, "layers": 1, "ff": 64}),
                      ("hope_phy", {"d_model": 32, "cms_levels": 3,
                                    "titans_hidden": 16, "backbone_layers": 1})]:
        m = build_model(name, D, L, cfg)
        assert m(x).shape == (B, L, D), name


def test_cms_periods_geometric():
    m = HopePhy(D, L, d_model=32, cms_levels=3, cms_base_period=4,
                backbone_layers=1, titans_hidden=16)
    periods = [b.period for b in m.cms.blocks]
    assert periods == [16, 4, 1]  # slow -> fast


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    m = build_model("hope_phy", D, L, {"d_model": 32, "cms_levels": 2,
                                       "titans_hidden": 16, "backbone_layers": 1,
                                       "dropout": 0.0})
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y = torch.randn(B, P, D), torch.randn(B, L, D) * 0.1
    l0 = None
    for _ in range(30):
        loss = nmse(m(x), y)
        l0 = l0 if l0 is not None else loss.detach().item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < l0


def test_ttt_engine_clocks_and_safety():
    torch.manual_seed(0)
    m = HopePhy(D, L, d_model=32, cms_levels=3, cms_base_period=4,
                backbone_layers=1, titans_hidden=16, dropout=0.0)
    eng = TTTEngine(m, TTTConfig(lr_slow=1e-4))
    x, y = torch.randn(2, P, D), torch.randn(2, L, D)
    slow_updates = 0
    for t in range(1, 33):
        rec = eng.observe(x, y)
        assert rec["surprise"] is not None
        if 16 in rec["updated_levels"]:
            slow_updates += 1
    # slow level (period 16) must update far less often than every step
    assert 0 < slow_updates <= 2
    # parameters must remain finite after adaptation
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_anchor_reset_restores_slow_level():
    m = HopePhy(D, L, d_model=32, cms_levels=2, backbone_layers=1,
                titans_hidden=16)
    m.cms.update_anchor(decay=0.0)  # anchor := current params
    before = [p.detach().clone() for p in m.cms.blocks[0].parameters()]
    with torch.no_grad():
        for p in m.cms.blocks[0].parameters():
            p.add_(1.0)
    m.cms.reset_slow_to_anchor()
    after = list(m.cms.blocks[0].parameters())
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(after, before))
