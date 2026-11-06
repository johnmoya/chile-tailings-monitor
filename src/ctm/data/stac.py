"""Sentinel-2 L2A access through STAC (default: Element84 Earth Search, public COGs).

For one deposit and one STAC item this module:
  1. reprojects the deposit polygon to the item's UTM CRS and buffers it;
  2. windowed-reads only that bbox from each band COG (no full-tile download);
  3. converts DN to surface reflectance with the asset's own ``raster:bands``
     scale/offset (Earth Search encodes the -0.1 offset of processing baseline
     >= 04.00 per item, so pre/post-2022 scenes end up on the same scale);
  4. resamples 20 m bands onto the 10 m grid of the first 10 m band;
  5. builds a validity mask from SCL.

Cloud-shadow rule (SCL = 3): dark water is frequently classified as shadow,
so class 3 is only treated as invalid when there are cloud pixels (SCL in
``scl_shadow_cloud_classes``) within ``shadow_cloud_radius_m`` of the bbox.
That needs SCL over a window larger than the bbox, read once per scene.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from ctm.utils.geo import buffered_bounds, polygon_mask, reproject_geometry

log = logging.getLogger(__name__)

SCL_CLASSES = list(range(12))


@dataclass
class Scene:
    """Clipped, co-registered multi-band scene for one deposit.

    ``image`` is NOT cloud-masked (NaN only where the COG has no data) so it can
    be exported for annotation; apply ``valid`` before any analysis.
    """

    image: np.ndarray            # float32 (C, H, W) reflectance, NaN where no data
    valid: np.ndarray            # bool (H, W): data in every band and SCL-clear
    scl: np.ndarray              # uint8 (H, W): SCL resampled (nearest) to the 10 m grid
    polygon_mask: np.ndarray     # bool (H, W): inside the deposit polygon
    bands: list[str]
    transform: Any               # affine.Affine of the 10 m grid
    crs: str
    item_id: str
    datetime: str
    meta: dict = field(default_factory=dict)

    @property
    def valid_fraction_in_polygon(self) -> float:
        n = int(self.polygon_mask.sum())
        return float((self.valid & self.polygon_mask).sum() / n) if n else 0.0

    def band(self, name: str) -> np.ndarray:
        return self.image[self.bands.index(name)]

    def masked_image(self) -> np.ndarray:
        out = self.image.copy()
        out[:, ~self.valid] = np.nan
        return out


# --------------------------------------------------------------------------- items


def item_crs(item: Item) -> str:
    code = item.properties.get("proj:code")
    if code:
        return str(code)
    epsg = item.properties.get("proj:epsg")
    if epsg:
        return f"EPSG:{epsg}"
    raise ValueError(f"item {item.id} has no proj:code / proj:epsg")


_TILE_RE = re.compile(r"(\d{2}[C-X][A-Z]{2})")


def item_tile(item: Item) -> str:
    """MGRS tile, e.g. ``19JBG``."""
    p = item.properties
    if p.get("grid:code"):
        return str(p["grid:code"]).removeprefix("MGRS-")
    if p.get("mgrs:utm_zone"):
        return f"{int(p['mgrs:utm_zone']):02d}{p['mgrs:latitude_band']}{p['mgrs:grid_square']}"
    if p.get("s2:mgrs_tile"):
        return str(p["s2:mgrs_tile"])
    m = _TILE_RE.search(item.id)
    if not m:
        raise ValueError(f"cannot determine MGRS tile of {item.id}")
    return m.group(1)


def dedupe_items(items: list[Item]) -> list[Item]:
    """Earth Search can hold reprocessed copies (``..._0_L2A``, ``..._1_L2A``) of the same
    acquisition; keep the highest sequence number per (satellite, MGRS tile, date)."""
    best: dict[str, tuple[int, Item]] = {}
    for it in items:
        key, seq = _acquisition_key(it.id)
        if key not in best or seq > best[key][0]:
            best[key] = (seq, it)
    return sorted((it for _, it in best.values()), key=lambda i: (i.datetime, i.id))


def _acquisition_key(item_id: str) -> tuple[str, int]:
    """``S2A_19JBG_20190129_1_L2A`` -> (``S2A_19JBG_20190129``, 1)."""
    parts = item_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return item_id, 0


def search_items(
    geometry_wgs84: BaseGeometry,
    stac_cfg: dict,
    date_start: str | None = None,
    date_end: str | None = None,
    max_items: int | None = None,
    max_cloud: float | None | bool = True,
    require_full_coverage: bool | None = None,
) -> list[Item]:
    """Items intersecting the geometry's bbox.

    ``max_cloud=True`` uses ``stac.max_scene_cloud_cover``; ``None``/``False`` disables
    the scene-level cloud filter (catalog building counts every acquisition).
    """
    client = Client.open(stac_cfg["endpoint"])
    start = date_start or stac_cfg["date_start"]
    end = date_end or stac_cfg["date_end"]
    if max_cloud is True:
        max_cloud = stac_cfg["max_scene_cloud_cover"]
    kwargs: dict[str, Any] = {}
    if max_cloud not in (None, False):
        kwargs["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
    search = client.search(
        collections=[stac_cfg["collection"]],
        intersects=geometry_wgs84.envelope,
        datetime=f"{start}/{end}",
        max_items=max_items,
        **kwargs,
    )
    items = dedupe_items(list(search.items()))
    full = stac_cfg.get("require_full_coverage", True) if require_full_coverage is None \
        else require_full_coverage
    if full:
        env = geometry_wgs84.envelope
        items = [it for it in items if shape(it.geometry).contains(env)]
    return items


# --------------------------------------------------------------------------- masking


def distance_to_bounds(xs: np.ndarray, ys: np.ndarray,
                       bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Euclidean distance from points to an axis-aligned rectangle (0 inside)."""
    minx, miny, maxx, maxy = bounds
    dx = np.maximum(np.maximum(minx - xs, 0.0), xs - maxx)
    dy = np.maximum(np.maximum(miny - ys, 0.0), ys - maxy)
    return np.hypot(dx, dy)


def clouds_near_bbox(scl_ext: np.ndarray, ext_transform: Affine,
                     bounds: tuple[float, float, float, float], radius_m: float,
                     cloud_classes: list[int]) -> bool:
    """True if any cloud pixel (centre) of ``scl_ext`` lies within ``radius_m`` of ``bounds``."""
    rows, cols = np.nonzero(np.isin(scl_ext, cloud_classes))
    if rows.size == 0:
        return False
    xs, ys = ext_transform * (cols + 0.5, rows + 0.5)
    return bool((distance_to_bounds(np.asarray(xs), np.asarray(ys), bounds) <= radius_m).any())


def scl_valid_mask(scl: np.ndarray, invalid_classes: list[int],
                   shadow_class: int | None = None, mask_shadow: bool = False) -> np.ndarray:
    invalid = list(invalid_classes)
    if mask_shadow and shadow_class is not None:
        invalid.append(shadow_class)
    return ~np.isin(scl, np.asarray(invalid, dtype=scl.dtype))


def scl_validity(scl: np.ndarray, scl_ext: np.ndarray, ext_transform: Affine,
                 bounds: tuple[float, float, float, float],
                 stac_cfg: dict) -> tuple[np.ndarray, bool]:
    """Validity of ``scl`` pixels under the configured rules, incl. the shadow rule.

    ``scl_ext``/``ext_transform`` cover ``bounds`` grown by ``shadow_cloud_radius_m``.
    Returns (valid mask, whether SCL=3 was treated as invalid).
    """
    shadow = clouds_near_bbox(scl_ext, ext_transform, bounds,
                              float(stac_cfg["shadow_cloud_radius_m"]),
                              stac_cfg["scl_shadow_cloud_classes"])
    valid = scl_valid_mask(scl, stac_cfg["scl_invalid_classes"],
                           stac_cfg["scl_shadow_class"], shadow)
    return valid, shadow


def scl_fractions(scl: np.ndarray, region: np.ndarray) -> dict[int, float]:
    """Fraction of ``region`` pixels in each SCL class 0..11."""
    n = int(region.sum())
    if n == 0:
        return {k: float("nan") for k in SCL_CLASSES}
    counts = np.bincount(scl[region].astype(np.int64), minlength=len(SCL_CLASSES))
    return {k: float(counts[k] / n) for k in SCL_CLASSES}


# --------------------------------------------------------------------------- reading


def _scale_offset(item: Item, asset_key: str, stac_cfg: dict) -> tuple[float, float]:
    bands = item.assets[asset_key].extra_fields.get("raster:bands") or [{}]
    scale = bands[0].get("scale", stac_cfg["default_scale"])
    offset = bands[0].get("offset", stac_cfg["default_offset"])
    return float(scale), float(offset)


GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def _read_clip(href: str, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    """Windowed read of ``bounds`` (asset CRS) from a COG; DN 0 (L2A nodata) -> NaN."""
    da = rioxarray.open_rasterio(href, masked=False, cache=False)
    da = da.rio.clip_box(*bounds).squeeze("band", drop=True).load()
    return da.where(da != 0).astype("float32").rio.write_nodata(np.nan)


def _expand(bounds, d):
    return (bounds[0] - d, bounds[1] - d, bounds[2] + d, bounds[3] + d)


def _deposit_bounds(item: Item, geometry_wgs84: BaseGeometry, stac_cfg: dict):
    crs = item_crs(item)
    geom_utm = reproject_geometry(geometry_wgs84, "EPSG:4326", crs)
    return crs, geom_utm, buffered_bounds(geom_utm, stac_cfg["bbox_buffer_m"])


@dataclass
class SclSummary:
    valid_fraction: float           # inside polygon, after the shadow rule
    fractions: dict[int, float]     # per SCL class, inside polygon
    shadow_masked: bool
    polygon_pixels: int


def summarize_scl(item: Item, geometry_wgs84: BaseGeometry, stac_cfg: dict) -> SclSummary:
    """Cheap per-scene SCL statistics at native 20 m (one COG read) for the catalog."""
    _, geom_utm, bounds = _deposit_bounds(item, geometry_wgs84, stac_cfg)
    radius = float(stac_cfg["shadow_cloud_radius_m"])
    with rasterio.Env(**GDAL_ENV):
        ext = _read_clip(item.assets[stac_cfg["asset_keys"]["SCL"]].href, _expand(bounds, radius))
    scl_ext = np.nan_to_num(ext.values, nan=0).astype("uint8")
    t = ext.rio.transform()
    valid, shadow = scl_validity(scl_ext, scl_ext, t, bounds, stac_cfg)
    poly = polygon_mask(geom_utm, t, scl_ext.shape)
    n = int(poly.sum())
    return SclSummary(
        valid_fraction=float((valid & poly).sum() / n) if n else 0.0,
        fractions=scl_fractions(scl_ext, poly), shadow_masked=shadow, polygon_pixels=n,
    )


def load_scene(item: Item, geometry_wgs84: BaseGeometry, stac_cfg: dict) -> Scene:
    """Windowed read + reflectance + 10 m resampling + SCL mask for one deposit."""
    crs, geom_utm, bounds = _deposit_bounds(item, geometry_wgs84, stac_cfg)
    keys = stac_cfg["asset_keys"]
    bands: list[str] = list(stac_cfg["bands"])
    resampling = Resampling[stac_cfg.get("resampling", "bilinear")]
    target_res = stac_cfg["target_resolution_m"]
    radius = float(stac_cfg["shadow_cloud_radius_m"])

    with rasterio.Env(**GDAL_ENV):
        raw = {b: _read_clip(item.assets[keys[b]].href, bounds) for b in bands}
        scl_ext = _read_clip(item.assets[keys["SCL"]].href, _expand(bounds, radius))

    ref_band = next(
        (b for b in bands if abs(abs(raw[b].rio.resolution()[0]) - target_res) < 1e-6), None
    )
    if ref_band is None:
        raise ValueError(f"no band at {target_res} m among {bands}")
    ref = raw[ref_band]

    layers = []
    for b in bands:
        da = raw[b]
        if da.shape != ref.shape or da.rio.transform() != ref.rio.transform():
            da = da.rio.reproject_match(ref, resampling=resampling)
        scale, offset = _scale_offset(item, keys[b], stac_cfg)
        layers.append(da.values.astype("float32") * scale + offset)
    image = np.stack(layers)

    ext_arr = np.nan_to_num(scl_ext.values, nan=0).astype("uint8")
    scl10 = scl_ext.rio.reproject_match(ref, resampling=Resampling.nearest).values
    scl10 = np.nan_to_num(scl10, nan=0).astype("uint8")  # NaN came from SCL==0 (no data)
    valid, shadow = scl_validity(scl10, ext_arr, scl_ext.rio.transform(), bounds, stac_cfg)
    valid &= np.isfinite(image).all(axis=0)

    transform = ref.rio.transform()
    poly = polygon_mask(geom_utm, transform, ref.shape)
    meta = {
        "collection": item.collection_id,
        "tile": item_tile(item),
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox_utm": list(bounds),
        "shadow_masked": shadow,
    }
    return Scene(
        image=image, valid=valid, scl=scl10, polygon_mask=poly, bands=bands,
        transform=transform, crs=crs, item_id=item.id,
        datetime=item.datetime.isoformat() if item.datetime else "", meta=meta,
    )
