"""Baseline: MNDWI = (B3 - B11) / (B3 + B11), Otsu threshold inside the deposit polygon.

The threshold is computed only from valid (cloud-free, finite) pixels inside
the polygon, then applied to those same pixels. Outside the polygon or on
invalid pixels the output is False and ``evaluated`` is False.

Otsu assumes a bimodal histogram. When there are too few pixels, or the values
are (near-)constant, the configured fixed threshold is used instead and
``fallback_used`` is set so downstream code can flag the scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu


@dataclass
class BaselineResult:
    water: np.ndarray        # bool (H, W), True = water (only where evaluated)
    evaluated: np.ndarray    # bool (H, W), valid & inside polygon
    mndwi: np.ndarray        # float32 (H, W), NaN where undefined
    threshold: float
    fallback_used: bool


def mndwi(green: np.ndarray, swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    green = green.astype("float32")
    swir = swir.astype("float32")
    denom = green + swir
    out = np.full(green.shape, np.nan, dtype="float32")
    ok = np.isfinite(green) & np.isfinite(swir) & (np.abs(denom) > eps)
    out[ok] = (green[ok] - swir[ok]) / denom[ok]
    return out


def otsu_or_fallback(values: np.ndarray, min_pixels: int, fallback: float) -> tuple[float, bool]:
    values = values[np.isfinite(values)]
    if values.size < min_pixels or np.ptp(values) < 1e-6:
        return float(fallback), True
    return float(threshold_otsu(values)), False


def segment(
    green: np.ndarray,
    swir: np.ndarray,
    polygon_mask: np.ndarray,
    valid: np.ndarray | None = None,
    min_valid_pixels: int = 200,
    fallback_threshold: float = 0.0,
    eps: float = 1e-6,
) -> BaselineResult:
    index = mndwi(green, swir, eps)
    evaluated = polygon_mask.astype(bool) & np.isfinite(index)
    if valid is not None:
        evaluated &= valid.astype(bool)
    thr, fallback = otsu_or_fallback(index[evaluated], min_valid_pixels, fallback_threshold)
    water = np.zeros(index.shape, dtype=bool)
    water[evaluated] = index[evaluated] > thr
    return BaselineResult(water, evaluated, index, thr, fallback)


def segment_scene(image: np.ndarray, bands: list[str], polygon_mask: np.ndarray,
                  baseline_cfg: dict, valid: np.ndarray | None = None) -> BaselineResult:
    """Convenience wrapper: pick green/SWIR by band name from a (C, H, W) stack."""
    return segment(
        image[bands.index(baseline_cfg["green_band"])],
        image[bands.index(baseline_cfg["swir_band"])],
        polygon_mask,
        valid=valid,
        min_valid_pixels=baseline_cfg["min_valid_pixels"],
        fallback_threshold=baseline_cfg["fallback_threshold"],
        eps=baseline_cfg["eps"],
    )
