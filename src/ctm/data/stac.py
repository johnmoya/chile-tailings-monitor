"""Sentinel-2 L2A access through STAC (default: Element84 Earth Search, public COGs).

For one deposit and one STAC item this module:
  1. reprojects the deposit polygon to the item's UTM CRS and buffers it;
  2. windowed-reads only that bbox from each band COG (no full-tile download);
  3. converts DN to surface reflectance with the asset's own ``raster:bands``
     scale/offset (Earth Search encodes the -0.1 offset of processing baseline
     >= 04.00 per item, so pre/post-2022 scenes end up on the same scale);
  4. resamples 20 m bands onto the 10 m grid of the first 10 m band;
  5. builds a validity mask from SCL (clouds, shadows, cirrus, snow, no data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from ctm.utils.geo import bounds_polygon, buffered_bounds, polygon_mask, reproject_geometry

log = logging.getLogger(__name__)


@dataclass
class Scene:
    """Clipped, co-registered multi-band scene for one deposit."""

    image: np.ndarray            # float32 (C, H, W) reflectance, NaN where invalid
    valid: np.ndarray            # bool (H, W): finite in every band and SCL-clear
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


def item_crs(item: Item) -> str:
    code = item.properties.get("proj:code")
    if code:
        return str(code)
    epsg = item.properties.get("proj:epsg")
    if epsg:
        return f"EPSG:{epsg}"
    raise ValueError(f"item {item.id} has no proj:code / proj:epsg")


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
) -> list[Item]:
    client = Client.open(stac_cfg["endpoint"])
    start = date_start or stac_cfg["date_start"]
    end = date_end or stac_cfg["date_end"]
    search = client.search(
        collections=[stac_cfg["collection"]],
        intersects=geometry_wgs84.envelope,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lte": stac_cfg["max_scene_cloud_cover"]}},
        max_items=max_items,
    )
    items = dedupe_items(list(search.items()))
    if stac_cfg.get("require_full_coverage", True):
        env = geometry_wgs84.envelope
        items = [it for it in items if shape(it.geometry).contains(env)]
    return items


def scl_valid_mask(scl: np.ndarray, invalid_classes: list[int]) -> np.ndarray:
    return ~np.isin(scl, np.asarray(invalid_classes, dtype=scl.dtype))


def _scale_offset(item: Item, asset_key: str, stac_cfg: dict) -> tuple[float, float]:
    bands = item.assets[asset_key].extra_fields.get("raster:bands") or [{}]
    scale = bands[0].get("scale", stac_cfg["default_scale"])
    offset = bands[0].get("offset", stac_cfg["default_offset"])
    return float(scale), float(offset)


GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def _read_clip(href: str, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    """Windowed read of ``bounds`` (asset CRS) from a COG; DN 0 (L2A nodata) -> NaN."""
    da = rioxarray.open_rasterio(href, masked=False, cache=False)
    da = da.rio.clip_box(*bounds).squeeze("band", drop=True).load()
    return da.where(da != 0).astype("float32").rio.write_nodata(np.nan)


def load_scene(item: Item, geometry_wgs84: BaseGeometry, stac_cfg: dict) -> Scene:
    """Windowed read + reflectance + 10 m resampling + SCL mask for one deposit."""
    crs = item_crs(item)
    geom_utm = reproject_geometry(geometry_wgs84, "EPSG:4326", crs)
    bounds = buffered_bounds(geom_utm, stac_cfg["bbox_buffer_m"])
    keys = stac_cfg["asset_keys"]
    bands: list[str] = list(stac_cfg["bands"])
    resampling = Resampling[stac_cfg.get("resampling", "bilinear")]
    target_res = stac_cfg["target_resolution_m"]

    with rasterio.Env(**GDAL_ENV):
        raw = {b: _read_clip(item.assets[keys[b]].href, bounds) for b in bands}
        scl = _read_clip(item.assets[keys["SCL"]].href, bounds)

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

    scl10 = scl.rio.reproject_match(ref, resampling=Resampling.nearest).values
    scl10 = np.nan_to_num(scl10, nan=0).astype("uint8")  # NaN came from SCL==0 (no data)
    valid = scl_valid_mask(scl10, stac_cfg["scl_invalid_classes"])
    valid &= np.isfinite(image).all(axis=0)
    image[:, ~valid] = np.nan

    transform = ref.rio.transform()
    poly = polygon_mask(geom_utm, transform, ref.shape)
    meta = {
        "collection": item.collection_id,
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox_utm": list(bounds),
        "footprint_contains_bbox": shape(item.geometry).contains(
            reproject_geometry(bounds_polygon(bounds), crs, "EPSG:4326")
        ),
    }
    return Scene(
        image=image, valid=valid, polygon_mask=poly, bands=bands, transform=transform, crs=crs,
        item_id=item.id, datetime=item.datetime.isoformat() if item.datetime else "", meta=meta,
    )
