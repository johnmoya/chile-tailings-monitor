"""Download (windowed) the scenes of a selection into data/scenes/<deposit_id>/<scene_id>.npz."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pystac import Item

from ctm.data.deposits import DepositRegistry
from ctm.data.scene_io import from_scene, save_scene
from ctm.data.stac import load_scene

log = logging.getLogger(__name__)


def load_item(href: str) -> Item:
    return Item.from_file(href)


def ingest(selection: pd.DataFrame, registry: DepositRegistry, cfg: dict, scenes_dir: Path,
           workers: int = 8) -> pd.DataFrame:
    """Fetch every row of ``selection`` (catalog rows). Existing files are skipped."""
    scenes_dir = Path(scenes_dir)

    def work(row) -> dict:
        out = scenes_dir / row.deposit_id / f"{row.scene_id}.npz"
        if out.exists():
            return {"deposit_id": row.deposit_id, "scene_id": row.scene_id, "status": "exists"}
        try:
            scene = load_scene(load_item(row.item_href), registry.geometry(row.deposit_id),
                               cfg["stac"])
            save_scene(out, from_scene(scene, row.deposit_id))
            return {"deposit_id": row.deposit_id, "scene_id": row.scene_id, "status": "ok"}
        except Exception as exc:
            log.warning("%s %s: %r", row.deposit_id, row.scene_id, exc)
            return {"deposit_id": row.deposit_id, "scene_id": row.scene_id,
                    "status": f"error: {exc!r}"[:300]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(work, selection.itertuples(index=False))))
