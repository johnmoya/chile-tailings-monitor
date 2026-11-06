"""Annotation kit export and validation.

Per scene, ``<annotation_dir>/<deposit_id>/<YYYY-MM-DD>/``:
  rgb.tif         uint8 x3, B4-B3-B2, one joint percentile stretch (keeps hue)
  falsecolor.tif  uint8 x3, B12-B8-B4, per-band percentile stretch
  mask.tif        uint8 x1, same grid, all 255 (0 = not water, 1 = water, 255 = ignore)
  polygon.geojson deposit outline in the scene CRS (for QGIS overlay)
  meta.json       scene id, item id, date, tile, CRS, cloud cover, valid fraction

Deliberately NOT exported: MNDWI, any baseline/model output, SCL. The
annotator must not be anchored on what the methods under test produce.
Imagery is NOT cloud-masked, so the annotator sees clouds and marks them 255.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import mapping, shape

from ctm.utils.geo import polygon_mask

RGB_BANDS = ("B04", "B03", "B02")
FALSE_COLOR_BANDS = ("B12", "B08", "B04")
MASK_VALUES = (0, 1, 255)
FORBIDDEN_NAMES = ("mndwi", "baseline", "otsu", "pred", "scl")


def stretch_uint8(stack: np.ndarray, region: np.ndarray, p_low: float, p_high: float,
                  joint: bool) -> np.ndarray:
    """Percentile stretch to 1..255 computed over ``region``; 0 is reserved for no data."""
    out = np.zeros(stack.shape, dtype="uint8")
    finite = np.isfinite(stack)
    sample_mask = region & finite.all(axis=0)

    def limits(arr):
        vals = arr[..., sample_mask]
        if vals.size == 0:
            vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(vals, [p_low, p_high])
        return float(lo), float(hi if hi > lo else lo + 1e-6)

    if joint:
        lo, hi = limits(stack)
        lims = [(lo, hi)] * stack.shape[0]
    else:
        lims = [limits(stack[i]) for i in range(stack.shape[0])]
    for i, (lo, hi) in enumerate(lims):
        scaled = np.clip((stack[i] - lo) / (hi - lo), 0, 1) * 254 + 1
        out[i] = np.where(finite[i], np.round(scaled), 0).astype("uint8")
    return out


def _profile(transform, crs, h, w, count, nodata):
    prof = {"driver": "GTiff", "height": h, "width": w, "count": count, "dtype": "uint8",
            "crs": crs, "transform": transform, "nodata": nodata, "compress": "deflate"}
    if h >= 256 and w >= 256:
        prof.update(tiled=True, blockxsize=256, blockysize=256)
    return prof


def write_tif(path: Path, arr: np.ndarray, transform, crs, nodata=None,
              descriptions: tuple[str, ...] | None = None) -> None:
    arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", **_profile(transform, crs, arr.shape[1], arr.shape[2],
                                             arr.shape[0], nodata)) as dst:
        dst.write(arr)
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)


def export_scene(scene, deposit_id: str, polygon_utm, out_root: Path, cfg: dict,
                 extra_meta: dict | None = None) -> Path:
    """Write the annotation folder for one :class:`ctm.data.stac.Scene`."""
    date = scene.datetime[:10]
    out = Path(out_root) / deposit_id / date
    out.mkdir(parents=True, exist_ok=True)
    p_lo, p_hi = cfg["annotation"]["stretch_percentiles"]
    idx = {b: scene.bands.index(b) for b in scene.bands}
    region = scene.polygon_mask
    rgb = stretch_uint8(scene.image[[idx[b] for b in RGB_BANDS]], region, p_lo, p_hi, True)
    fc = stretch_uint8(scene.image[[idx[b] for b in FALSE_COLOR_BANDS]], region, p_lo, p_hi,
                       False)
    write_tif(out / "rgb.tif", rgb, scene.transform, scene.crs, 0, RGB_BANDS)
    write_tif(out / "falsecolor.tif", fc, scene.transform, scene.crs, 0, FALSE_COLOR_BANDS)
    mask_path = out / "mask.tif"
    if not mask_path.exists():  # never overwrite work in progress
        template = np.full(scene.polygon_mask.shape, 255, dtype="uint8")
        write_tif(mask_path, template, scene.transform, scene.crs, None,
                  ("0=not water, 1=water, 255=ignore",))
    (out / "polygon.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": scene.crs}},
        "features": [{"type": "Feature", "properties": {"deposit_id": deposit_id},
                      "geometry": mapping(polygon_utm)}]}))
    meta = {"deposit_id": deposit_id, "date": date, "item_id": scene.item_id,
            "crs": scene.crs, "transform": list(scene.transform)[:6],
            "shape": list(scene.polygon_mask.shape),
            "tile": scene.meta.get("tile"),
            "processing_baseline": scene.meta.get("processing_baseline"),
            "scene_cloud_cover": scene.meta.get("scene_cloud_cover"),
            **(extra_meta or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- validation


@dataclass
class MaskReport:
    path: Path
    status: str                    # ok | pending | incomplete | error
    coverage: float = float("nan")  # labelled fraction of polygon pixels
    water_fraction: float = float("nan")  # of labelled polygon pixels
    problems: list[str] = field(default_factory=list)


def validate_scene_dir(d: Path, min_coverage: float) -> MaskReport:
    d = Path(d)
    rep = MaskReport(d, "error")
    for name in ("rgb.tif", "falsecolor.tif", "mask.tif", "polygon.geojson"):
        if not (d / name).exists():
            rep.problems.append(f"missing {name}")
    forbidden = [p.name for p in d.iterdir() if any(f in p.name.lower() for f in FORBIDDEN_NAMES)]
    if forbidden:
        rep.problems.append(f"forbidden files present: {forbidden}")
    if rep.problems and "missing mask.tif" in rep.problems:
        return rep
    with rasterio.open(d / "mask.tif") as m:
        mask, mprof = m.read(), m.profile
    if mask.shape[0] != 1:
        rep.problems.append(f"mask must have 1 band, has {mask.shape[0]}")
    mask = mask[0]
    if mprof["dtype"] != "uint8":
        rep.problems.append(f"mask dtype {mprof['dtype']} != uint8")
    for ref in ("rgb.tif", "falsecolor.tif"):
        if (d / ref).exists():
            with rasterio.open(d / ref) as r:
                if (r.crs, r.transform, r.height, r.width) != \
                        (mprof["crs"], mprof["transform"], mprof["height"], mprof["width"]):
                    rep.problems.append(f"mask grid differs from {ref}")
    bad = sorted(set(np.unique(mask).tolist()) - set(MASK_VALUES))
    if bad:
        rep.problems.append(f"invalid mask values {bad[:10]}")
    if (d / "polygon.geojson").exists():
        fc = json.loads((d / "polygon.geojson").read_text())
        poly = shape(fc["features"][0]["geometry"])
        region = polygon_mask(poly, mprof["transform"], mask.shape)
    else:
        region = np.ones(mask.shape, bool)
    n = int(region.sum())
    labelled = region & (mask != 255)
    rep.coverage = float(labelled.sum() / n) if n else float("nan")
    rep.water_fraction = float((mask[labelled] == 1).mean()) if labelled.any() else float("nan")
    if rep.problems:
        return rep
    if not labelled.any():
        rep.status = "pending"
    elif rep.coverage < min_coverage:
        rep.status = "incomplete"
    else:
        rep.status = "ok"
    return rep


def scene_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/mask.tif")) + \
        sorted(p for p in Path(root).glob("*/*") if p.is_dir() and not (p / "mask.tif").exists())
