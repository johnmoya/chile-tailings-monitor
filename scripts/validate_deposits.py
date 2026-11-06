"""Validate configs/deposits.geojson (schema, no placeholders, all splits used)
and configs/crest_lines.geojson (schema; may still be empty).

Exit 0 when valid, 1 otherwise. Usage: ``uv run python scripts/validate_deposits.py``
"""

from __future__ import annotations

import argparse
import sys

from ctm.data.crest import load_crests
from ctm.data.deposits import SPLITS, DepositSchemaError, load_deposits
from ctm.utils.config import load_config, resolve_path
from ctm.utils.geo import reproject_geometry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    path = resolve_path(cfg, "deposits")
    try:
        reg = load_deposits(path, allow_placeholders=False)
    except DepositSchemaError as exc:
        print(f"INVALID {path}: {exc}", file=sys.stderr)
        return 1
    problems = []
    if not reg.ids:
        problems.append("no deposits")
    print(f"{path}: {len(reg.ids)} deposits")
    for split in SPLITS:
        ids = list(reg.gdf.loc[reg.gdf["split"] == split, "id"])
        print(f"  {split:5s} {len(ids):2d}  {', '.join(ids)}")
        if not ids:
            problems.append(f"split {split!r} is empty")
    for _, row in reg.gdf.iterrows():
        utm = reproject_geometry(row.geometry, "EPSG:4326", "EPSG:32719")
        print(f"  {row['id']:24s} {row['region']:16s} area {utm.area / 1e6:7.2f} km2")
    try:
        crests = load_crests(resolve_path(cfg, "crest_lines"), reg)
        missing = sorted(set(reg.ids) - set(crests.deposit_ids))
        print(f"crest lines: {len(crests.deposit_ids)} loaded, {len(missing)} pending")
    except DepositSchemaError as exc:
        problems.append(f"crest_lines: {exc}")
    if problems:
        print("INVALID: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
