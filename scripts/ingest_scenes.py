"""Download the scenes of a selection (train sample or series) as .npz.

Usage: ``uv run python scripts/ingest_scenes.py --selection train [--workers 8]``
Resumable: existing files are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ctm.data.deposits import load_deposits
from ctm.data.ingest import ingest
from ctm.utils.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--selection", choices=["train", "series"], required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    reg = load_deposits(resolve_path(cfg, "deposits"), allow_placeholders=False)
    sel = pd.read_parquet(resolve_path(cfg, "samples_dir") / f"{args.selection}_scenes.parquet")
    res = ingest(sel, reg, cfg, resolve_path(cfg, "scenes_dir"), args.workers)
    print(res.groupby(["deposit_id", res["status"].str.split(":").str[0]]).size()
          .unstack(fill_value=0).to_string())
    return 0 if not res["status"].str.startswith("error").any() else 1


if __name__ == "__main__":
    sys.exit(main())
