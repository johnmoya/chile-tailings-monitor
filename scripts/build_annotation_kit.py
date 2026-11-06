"""Build the annotation kit for val and test deposits.

Selects ``annotation.scenes_per_deposit`` scenes per deposit from the catalog
(stratified by season and year, fixed seed, mandatory Jul–Aug 2026 windows)
and exports RGB, false colour and an empty mask template per scene.
No MNDWI, baseline or SCL output is exported.

Reading test-deposit imagery here is intentional: the manual reference masks
are what the single OOS run will be scored against. No model or baseline
output is computed on these scenes.

Usage: ``uv run python scripts/build_annotation_kit.py``
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ctm.data.deposits import load_deposits
from ctm.data.ingest import load_item
from ctm.data.sampling import annotation_selection
from ctm.data.stac import item_crs, load_scene
from ctm.labels.annotation import export_scene
from ctm.utils.config import load_config, resolve_path
from ctm.utils.geo import reproject_geometry

README = """# Annotation kit

One folder per scene: `<deposit_id>/<YYYY-MM-DD>/`.

* `rgb.tif` (B4-B3-B2) and `falsecolor.tif` (B12-B8-B4): 10 m, uint8, display only.
* `mask.tif`: paint it on the SAME grid. Values: 0 = not water, 1 = water,
  255 = ignore (cloud, cloud shadow over the pond, no data, cannot tell).
  The template starts all 255. Label every pixel inside `polygon.geojson`.
* Water = free supernatant water, including shallow/turbid water. Wet beach
  without standing water = 0.
* Do not open any MNDWI / baseline / model output while annotating.

Check progress with `uv run python scripts/validate_annotations.py`.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--select-only", action="store_true", help="write the manifest only")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    a = cfg["annotation"]
    reg = load_deposits(resolve_path(cfg, "deposits"), allow_placeholders=False)
    catalog = pd.read_parquet(resolve_path(cfg, "catalog"))
    ids = [d for s in a["splits"] for d in reg.ids_for(s, allow_test=(s == "test"))]

    picks = []
    for did in ids:
        sel = annotation_selection(catalog[catalog["deposit_id"] == did],
                                   a["scenes_per_deposit"], a["seed"], did, a["seasons"],
                                   a["min_valid_fraction"], a["mandatory"],
                                   a["mandatory_min_valid_fraction"])
        if len(sel) < a["scenes_per_deposit"]:
            logging.warning("%s: only %d candidate scenes", did, len(sel))
        picks.append(sel)
    manifest = pd.concat(picks, ignore_index=True)
    root = resolve_path(cfg, "annotation_dir")
    root.mkdir(parents=True, exist_ok=True)
    manifest.drop(columns=[c for c in manifest.columns if c.startswith("scl_")]).to_csv(
        root / "manifest.csv", index=False)
    (root / "README.md").write_text(README)
    if args.select_only:
        print(manifest[["deposit_id", "date", "season", "selection_reason"]].to_string())
        return 0

    def work(row):
        item = load_item(row.item_href)
        geom = reg.geometry(row.deposit_id)
        scene = load_scene(item, geom, cfg["stac"])
        poly = reproject_geometry(geom, "EPSG:4326", item_crs(item))
        return export_scene(scene, row.deposit_id, poly, root, cfg,
                            {"scene_id": row.scene_id, "season": row.season,
                             "selection_reason": row.selection_reason,
                             "valid_fraction_scl": row.valid_fraction})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        dirs = list(pool.map(work, manifest.itertuples(index=False)))
    print(manifest.groupby("deposit_id").size().to_string())
    print(f"{len(dirs)} scenes exported under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
