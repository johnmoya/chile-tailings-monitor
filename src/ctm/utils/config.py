"""YAML configuration loading with dotted-key overrides and repo-relative paths."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_override(text: str) -> tuple[str, Any]:
    """``"train.epochs=3"`` -> ``("train.epochs", 3)``; the value is parsed as YAML."""
    if "=" not in text:
        raise ValueError(f"override must look like key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
) -> dict:
    """Load a YAML config. Other files may ``inherit: default.yaml`` (path relative to them).

    The returned dict carries ``_root`` (repo root used to resolve relative paths) and
    ``_source`` (the config file actually read).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    path = path.resolve()
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        base.pop("_source", None)
        base.pop("_root", None)
        cfg = _deep_merge(base, cfg)
    if overrides:
        items = overrides.items() if isinstance(overrides, dict) else map(parse_override, overrides)
        for key, value in items:
            set_dotted(cfg, key, value)
    cfg.setdefault("_root", str(path.parent.parent))
    cfg["_source"] = str(path)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve ``paths.<key>`` against the config's root unless already absolute."""
    raw = get_dotted(cfg, f"paths.{key}")
    if raw is None:
        raise KeyError(f"paths.{key} is not set in config {cfg.get('_source')}")
    p = Path(raw).expanduser()
    return p if p.is_absolute() else Path(cfg["_root"]) / p
