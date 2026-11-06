"""Test helpers. All deposits here are FICTITIOUS squares, not real facilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ctm.data.scene_io import StoredScene, save_scene


def fake_square(lon: float, lat: float, d: float = 0.01) -> dict:
    return {"type": "Polygon", "coordinates": [[
        [lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]}


def write_deposits(path: Path, rows: list[dict]) -> Path:
    feats = []
    for i, r in enumerate(rows):
        props = {"nombre": f"ficticio {i}", "region": "test", **r}
        geom = props.pop("geometry", fake_square(-70.0 + i * 0.05, -30.0))
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path


def synthetic_eval_scene(bands: list[str], deposit_id: str, scene_id: str, size: int = 96,
                         seed: int = 0) -> StoredScene:
    rng = np.random.default_rng(seed)
    water = np.zeros((size, size), bool)
    water[20:60, 25:70] = True
    image = rng.normal(0.15, 0.01, (len(bands), size, size)).astype("float32")
    g, s = bands.index("B03"), bands.index("B11")
    image[g] = np.where(water, 0.10, 0.08)
    image[s] = np.where(water, 0.02, 0.25)
    poly = np.zeros_like(water)
    poly[5:90, 5:90] = True
    image[:, :4, :] = np.nan  # cloud strip
    ref = water.astype("uint8")
    return StoredScene(image=image, polygon_mask=poly, bands=bands, reference=ref,
                       meta={"deposit_id": deposit_id, "scene_id": scene_id,
                             "pixel_area_m2": 100.0})


def fake_catalog(deposit_ids: list[str], start="2019-01-01", end="2026-09-30", every=5,
                 seed=0, split="val"):
    """Catalog-shaped DataFrame with random but plausible SCL statistics."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for did in deposit_ids:
        for date in pd.date_range(start, end, freq=f"{every}D"):
            cloud = float(rng.uniform(0, 60))
            frac = np.zeros(12)
            frac[5] = 0.7
            frac[6] = 0.3
            if rng.random() < cloud / 100:
                frac[9] = rng.uniform(0, 0.5)
                frac[5] -= frac[9]
            row = {"deposit_id": did, "split": split, "date": date,
                   "scene_id": f"{date:%Y%m%d}_S2A", "item_id": f"S2A_19JBG_{date:%Y%m%d}_0_L2A",
                   "item_href": "", "tile": "19JBG", "platform": "sentinel-2a",
                   "processing_baseline": "05.10", "scene_cloud_cover": cloud,
                   "valid_fraction": float(1 - frac[9]), "shadow_masked": False,
                   "polygon_pixels": 1000}
            row.update({f"scl_{k}": float(frac[k]) for k in range(12)})
            rows.append(row)
    return pd.DataFrame(rows)


def write_eval_scene(scenes_dir: Path, scene: StoredScene) -> Path:
    return save_scene(scenes_dir / scene.meta["deposit_id"] / f"{scene.meta['scene_id']}.npz",
                      scene)
