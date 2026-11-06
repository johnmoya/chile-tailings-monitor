"""Deposit-level physical metrics derived from a water mask (projected CRS, metres).

* pond area (m^2)
* beach fraction: inside the polygon eroded by ``inset_m`` (keeps the dam faces and
  the polygon's edge pixels out), the share of valid pixels that are NOT water.
* minimum pond–crest distance (m): distance between the pond (union of its pixel
  squares, so the pond edge, not pixel centres) and the crest line. 0 if they touch.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ctm.utils.geo import polygon_mask


def pixel_area_m2(transform: Affine) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d)


def pond_area_m2(water: np.ndarray, transform: Affine, valid: np.ndarray | None = None) -> float:
    w = water.astype(bool) if valid is None else water.astype(bool) & valid.astype(bool)
    return float(w.sum() * pixel_area_m2(transform))


def inset_mask(polygon_utm: BaseGeometry, inset_m: float, transform: Affine,
               shape_hw: tuple[int, int]) -> np.ndarray:
    inner = polygon_utm.buffer(-inset_m)
    if inner.is_empty:
        return np.zeros(shape_hw, dtype=bool)
    return polygon_mask(inner, transform, shape_hw)


def beach_fraction(water: np.ndarray, inset: np.ndarray, valid: np.ndarray | None = None,
                   min_valid_fraction: float = 0.8) -> float:
    """Share of valid inset pixels that are not water; NaN if too few valid pixels."""
    inset = inset.astype(bool)
    n = int(inset.sum())
    if n == 0:
        return float("nan")
    v = inset if valid is None else inset & valid.astype(bool)
    if v.sum() / n < min_valid_fraction:
        return float("nan")
    return float((v & ~water.astype(bool)).sum() / v.sum())


def pond_geometry(water: np.ndarray, transform: Affine) -> BaseGeometry | None:
    w = water.astype("uint8")
    if not w.any():
        return None
    polys = [shape(g) for g, val in shapes(w, mask=w.astype(bool), transform=transform)
             if val == 1]
    return unary_union(polys)


def pond_crest_distance_m(water: np.ndarray, transform: Affine,
                          crest_utm: BaseGeometry) -> float:
    """Minimum distance (m) between the pond and the crest; NaN if there is no water."""
    pond = pond_geometry(water, transform)
    if pond is None:
        return float("nan")
    return float(pond.distance(crest_utm))
