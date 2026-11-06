"""Small geometry helpers shared by data, baseline and labels."""

from __future__ import annotations

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry


def reproject_geometry(geom: BaseGeometry, src_crs: str, dst_crs: str) -> BaseGeometry:
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely.transform(geom, tf.transform, interleaved=False)


def buffered_bounds(geom: BaseGeometry, buffer: float) -> tuple[float, float, float, float]:
    """Bounds of ``geom`` grown by ``buffer`` (in the geometry's own units)."""
    return geom.buffer(buffer).bounds


def bounds_polygon(bounds: tuple[float, float, float, float]) -> BaseGeometry:
    return box(*bounds)


def polygon_mask(
    geom: BaseGeometry, transform: Affine, shape: tuple[int, int], all_touched: bool = False
) -> np.ndarray:
    """Boolean raster, True inside ``geom`` (geometry in the raster's CRS)."""
    return geometry_mask(
        [mapping(geom)], out_shape=shape, transform=transform, invert=True,
        all_touched=all_touched,
    )
