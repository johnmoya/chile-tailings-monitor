"""Persist a :class:`~ctm.data.stac.Scene` (or an evaluation scene) as a single .npz."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StoredScene:
    image: np.ndarray            # float32 (C, H, W), NaN where invalid
    polygon_mask: np.ndarray     # bool (H, W)
    bands: list[str]
    meta: dict                   # deposit_id, item_id, datetime, crs, transform, ...
    reference: np.ndarray | None = None   # uint8 (H, W), manual mask (val/test scoring)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.image).all(axis=0)


def save_scene(path: str | Path, scene: StoredScene) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"image": scene.image.astype("float32"),
              "polygon_mask": scene.polygon_mask.astype(bool)}
    if scene.reference is not None:
        arrays["reference"] = scene.reference.astype("uint8")
    meta = dict(scene.meta, bands=list(scene.bands))
    np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrays)
    return path


def load_scene_npz(path: str | Path) -> StoredScene:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        ref = z["reference"] if "reference" in z.files else None
        return StoredScene(
            image=z["image"], polygon_mask=z["polygon_mask"].astype(bool),
            bands=meta.pop("bands"), meta=meta, reference=ref,
        )
