"""Persist a :class:`~ctm.data.stac.Scene` (or an evaluation scene) as a single .npz."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StoredScene:
    image: np.ndarray            # float32 (C, H, W), NaN where no data (NOT cloud-masked)
    polygon_mask: np.ndarray     # bool (H, W)
    bands: list[str]
    meta: dict                   # deposit_id, scene_id, item_id, datetime, crs, transform, ...
    reference: np.ndarray | None = None    # uint8 (H, W), manual mask (val/test scoring)
    valid_mask: np.ndarray | None = None   # bool (H, W), SCL rules applied
    scl: np.ndarray | None = None          # uint8 (H, W)

    @property
    def valid(self) -> np.ndarray:
        finite = np.isfinite(self.image).all(axis=0)
        return finite if self.valid_mask is None else finite & self.valid_mask


def from_scene(scene, deposit_id: str) -> StoredScene:
    """Build a StoredScene from a :class:`ctm.data.stac.Scene`."""
    meta = dict(scene.meta, deposit_id=deposit_id, scene_id=scene_id_of(scene.datetime,
                scene.item_id), item_id=scene.item_id, datetime=scene.datetime,
                crs=scene.crs, transform=list(scene.transform)[:6], pixel_area_m2=abs(
                    scene.transform.a * scene.transform.e))
    return StoredScene(image=scene.image, polygon_mask=scene.polygon_mask, bands=scene.bands,
                       meta=meta, valid_mask=scene.valid, scl=scene.scl)


def scene_id_of(datetime_iso: str, item_id: str) -> str:
    """``2019-01-29`` + item -> ``20190129_S2A`` (date first so files sort by time)."""
    return f"{datetime_iso[:10].replace('-', '')}_{item_id.split('_')[0]}"


def save_scene(path: str | Path, scene: StoredScene) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"image": scene.image.astype("float32"),
              "polygon_mask": scene.polygon_mask.astype(bool)}
    if scene.reference is not None:
        arrays["reference"] = scene.reference.astype("uint8")
    if scene.valid_mask is not None:
        arrays["valid_mask"] = scene.valid_mask.astype(bool)
    if scene.scl is not None:
        arrays["scl"] = scene.scl.astype("uint8")
    meta = dict(scene.meta, bands=list(scene.bands))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta, default=str), **arrays)
    tmp.replace(path)
    return path


def load_scene_npz(path: str | Path) -> StoredScene:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        opt = {k: z[k] for k in ("reference", "valid_mask", "scl") if k in z.files}
        valid = opt.get("valid_mask")
        return StoredScene(
            image=z["image"], polygon_mask=z["polygon_mask"].astype(bool),
            bands=meta.pop("bands"), meta=meta, reference=opt.get("reference"),
            valid_mask=None if valid is None else valid.astype(bool), scl=opt.get("scl"),
        )
