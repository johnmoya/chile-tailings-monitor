"""Smoke test: 1 epoch of U-Net training on synthetic tiles (GPU by default).

Usage: ``uv run python scripts/train_smoke.py [--n-train 64] [key.path=value ...]``
Prints per-epoch metrics and peak VRAM. Uses random encoder init (no download).
"""

from __future__ import annotations

import argparse
import json
import sys

from ctm.data.dataset import SyntheticWaterDataset
from ctm.train.loop import train
from ctm.utils.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.device=cpu")
    args = p.parse_args(argv)

    # Smoke defaults first; user overrides win because they are applied last.
    base = ["train.epochs=1", "model.encoder_weights=null", "train.num_workers=0"]
    cfg = load_config(args.config, overrides=base + args.overrides)

    size = cfg["tiling"]["tile_size"]
    bands = cfg["stac"]["bands"]
    kw = dict(size=size, channels=cfg["model"]["in_channels"],
              ignore_index=cfg["labels"]["ignore_index"],
              green_idx=bands.index(cfg["baseline"]["green_band"]),
              swir_idx=bands.index(cfg["baseline"]["swir_band"]))
    train_ds = SyntheticWaterDataset(args.n_train, seed=cfg["seed"], **kw)
    val_ds = SyntheticWaterDataset(args.n_val, seed=cfg["seed"] + 1, **kw)

    run_dir = resolve_path(cfg, "runs_dir") / "smoke"
    if (run_dir / "metrics.jsonl").exists():
        (run_dir / "metrics.jsonl").unlink()
    res = train(cfg, train_ds, val_ds, run_dir)
    for rec in res.history:
        print(json.dumps(rec))
    if res.peak_vram_mib is not None:
        print(f"peak VRAM allocated: {res.peak_vram_mib:.0f} MiB "
              f"(batch {cfg['train']['batch_size']}, {size}x{size}, "
              f"amp={cfg['train']['amp']} {cfg['train']['amp_dtype']})")
    print(f"checkpoints: {res.last_checkpoint}, {res.best_checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
