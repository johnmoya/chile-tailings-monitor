"""Scene catalog: one fixed MGRS tile per deposit, one row per acquisition on that tile.

Columns of ``data/catalog.parquet``:
  deposit_id, split, scene_id, item_id, item_href, date, tile, platform,
  processing_baseline, scene_cloud_cover (eo:cloud_cover of the whole tile, %),
  valid_fraction (inside the polygon, after SCL rules incl. the shadow rule),
  shadow_masked, polygon_pixels (20 m), scl_0 .. scl_11 (class fractions in polygon).

The catalog is split-agnostic metadata (SCL statistics only, no reflectance),
so it covers test deposits too; nothing here feeds a model.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pystac import Item

from ctm.data.deposits import DepositRegistry
from ctm.data.scene_io import scene_id_of
from ctm.data.stac import SCL_CLASSES, SclSummary, item_tile, search_items, summarize_scl

log = logging.getLogger(__name__)

SCL_COLS = [f"scl_{k}" for k in SCL_CLASSES]


def choose_tile(items: Iterable[Item]) -> tuple[str, dict[str, int]]:
    """Tile with most acquisitions among the given (already full-coverage) items.

    Ties are broken alphabetically so the choice is deterministic.
    """
    counts = Counter(item_tile(it) for it in items)
    if not counts:
        raise ValueError("no item fully covers the deposit bbox")
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best, dict(counts)


def _row(deposit_id: str, split: str, item: Item, s: SclSummary) -> dict:
    dt = item.datetime.isoformat() if item.datetime else ""
    row = {
        "deposit_id": deposit_id,
        "split": split,
        "scene_id": scene_id_of(dt, item.id),
        "item_id": item.id,
        "item_href": item.get_self_href(),
        "date": pd.Timestamp(dt[:10]),
        "tile": item_tile(item),
        "platform": item.properties.get("platform", item.id.split("_")[0]),
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
        "valid_fraction": s.valid_fraction,
        "shadow_masked": s.shadow_masked,
        "polygon_pixels": s.polygon_pixels,
    }
    row.update({f"scl_{k}": s.fractions[k] for k in SCL_CLASSES})
    return row


def catalog_deposit(
    deposit_id: str,
    split: str,
    geometry,
    cfg: dict,
    search_fn: Callable = search_items,
    summarize_fn: Callable = summarize_scl,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stac_cfg = cfg["stac"]
    items = search_fn(geometry, stac_cfg, max_cloud=None, require_full_coverage=True)
    tile, counts = choose_tile(items)
    on_tile = [it for it in items if item_tile(it) == tile]
    log.info("%s: tile %s (%d scenes; candidates %s)", deposit_id, tile, len(on_tile), counts)

    def work(it: Item) -> dict | None:
        try:
            return _row(deposit_id, split, it, summarize_fn(it, geometry, stac_cfg))
        except Exception as exc:  # network hiccups: keep going, report
            log.warning("%s %s: %r", deposit_id, it.id, exc)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(work, on_tile) if r is not None]
    failed = len(on_tile) - len(rows)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
    df.attrs["failed"] = failed
    return df, counts


def build_catalog(
    registry: DepositRegistry,
    cfg: dict,
    parts_dir: str | Path,
    search_fn: Callable = search_items,
    summarize_fn: Callable = summarize_scl,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Catalog all non-placeholder deposits, caching one parquet per deposit."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    frames, tiles = [], []
    reg = registry.real()
    for _, dep in reg.gdf.iterrows():
        did = dep["id"]
        part, tpart = parts_dir / f"{did}.parquet", parts_dir / f"{did}.tiles.parquet"
        if part.exists() and tpart.exists() and not refresh:
            frames.append(pd.read_parquet(part))
            tiles.append(pd.read_parquet(tpart))
            continue
        df, counts = catalog_deposit(did, dep["split"], dep.geometry, cfg, search_fn,
                                     summarize_fn, cfg["catalog"]["workers"])
        chosen = df["tile"].iloc[0] if len(df) else None
        tdf = pd.DataFrame([{"deposit_id": did, "tile": t, "n_scenes": n,
                             "chosen": t == chosen} for t, n in counts.items()])
        if df.attrs.get("failed"):
            log.warning("%s: %d scenes failed; part not cached", did, df.attrs["failed"])
        else:
            df.to_parquet(part, index=False)
            tdf.to_parquet(tpart, index=False)
        frames.append(df)
        tiles.append(tdf)
    catalog = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return catalog, pd.concat(tiles, ignore_index=True) if tiles else pd.DataFrame()
