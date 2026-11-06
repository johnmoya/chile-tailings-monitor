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


def write_eval_scene(scenes_dir: Path, scene: StoredScene) -> Path:
    return save_scene(scenes_dir / scene.meta["deposit_id"] / f"{scene.meta['scene_id']}.npz",
                      scene)
