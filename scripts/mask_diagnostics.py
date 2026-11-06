"""Write reports/m1_mask_diagnostics.md from data/catalog.parquet.

Usage: ``uv run python scripts/mask_diagnostics.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ctm.data.diagnostics import render_markdown
from ctm.utils.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    catalog = pd.read_parquet(resolve_path(cfg, "catalog"))
    out = resolve_path(cfg, "reports_dir") / "m1_mask_diagnostics.md" if args.out is None \
        else Path(args.out)
    md = render_markdown(catalog, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
