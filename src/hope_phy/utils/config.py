"""Composable YAML config system.

Usage from CLI:
    python scripts/train.py model=lstm data=synthetic train.epochs=5

- ``base.yaml`` is always loaded first.
- ``group=name`` loads ``configs/<group>/<name>.yaml`` under key ``<group>``.
- ``a.b.c=value`` sets a dotted override (YAML-parsed, so numbers/bools work).

Every resolved config is saved alongside the run for exact reproducibility.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_GROUPS = ("data", "model", "train", "ttt")


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(config_dir: str | Path, overrides: list[str] | None = None) -> dict:
    config_dir = Path(config_dir)
    cfg: dict = yaml.safe_load((config_dir / "base.yaml").read_text()) or {}

    group_choice = {g: cfg.get(g, {}).get("_default") for g in CONFIG_GROUPS}
    dotted: list[tuple[str, Any]] = []

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be key=value")
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        if key in CONFIG_GROUPS:
            group_choice[key] = val
        else:
            dotted.append((key, val))

    for group, choice in group_choice.items():
        if choice is None:
            continue
        path = config_dir / group / f"{choice}.yaml"
        sub = yaml.safe_load(path.read_text()) or {}
        cfg.setdefault(group, {})
        _deep_update(cfg[group], sub)
        cfg[group]["_name"] = choice

    for key, val in dotted:
        _set_dotted(cfg, key, val)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
