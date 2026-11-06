import numpy as np
import pytest

from ctm.baseline.mndwi_otsu import mndwi, segment, segment_scene
from ctm.metrics.segmentation import iou


def synthetic_pond(shape=(120, 160), seed=0, noise=0.01):
    """Green/SWIR with a known rectangular pond. Water: MNDWI ~ +0.7, land: ~ -0.5."""
    rng = np.random.default_rng(seed)
    water = np.zeros(shape, bool)
    water[30:80, 40:110] = True
    green = np.where(water, 0.10, 0.08) + rng.normal(0, noise, shape)
    swir = np.where(water, 0.02, 0.25) + rng.normal(0, noise / 4, shape)
    return green.astype("float32"), np.clip(swir, 1e-3, None).astype("float32"), water


def test_mndwi_formula_and_nan_handling():
    g = np.array([0.3, 0.1, 0.0, np.nan], dtype="float32")
    s = np.array([0.1, 0.3, 0.0, 0.2], dtype="float32")
    out = mndwi(g, s)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(-0.5)
    assert np.isnan(out[2]) and np.isnan(out[3])   # zero denominator, NaN input


def test_known_water_gives_iou_close_to_one():
    green, swir, water = synthetic_pond()
    poly = np.ones_like(water)
    res = segment(green, swir, poly)
    assert not res.fallback_used
    assert -0.5 < res.threshold < 0.7
    assert iou(res.water, water, res.evaluated) > 0.99


def test_threshold_uses_only_pixels_inside_polygon():
    green, swir, water = synthetic_pond()
    poly = np.zeros_like(water)
    poly[20:100, 20:140] = True
    # Outside the polygon put extreme "water" values that would drag a global Otsu.
    green[~poly], swir[~poly] = 0.5, 0.001
    res = segment(green, swir, poly)
    assert not res.water[~poly].any()
    assert not res.evaluated[~poly].any()
    assert iou(res.water, water, poly) > 0.99


def test_invalid_pixels_are_not_evaluated():
    green, swir, water = synthetic_pond()
    valid = np.ones_like(water)
    valid[:, :20] = False
    green[:, :10] = np.nan
    res = segment(green, swir, np.ones_like(water), valid=valid)
    assert not res.evaluated[:, :20].any()
    assert iou(res.water, water, res.evaluated) > 0.99


def test_fallback_on_too_few_pixels_or_constant_values():
    green, swir, _ = synthetic_pond()
    poly = np.zeros(green.shape, bool)
    poly[0:5, 0:5] = True
    res = segment(green, swir, poly, min_valid_pixels=200, fallback_threshold=0.0)
    assert res.fallback_used and res.threshold == 0.0
    const = segment(np.full((50, 50), 0.1), np.full((50, 50), 0.3), np.ones((50, 50), bool))
    assert const.fallback_used and not const.water.any()


def test_segment_scene_picks_bands_by_name(cfg):
    green, swir, water = synthetic_pond()
    bands = cfg["stac"]["bands"]
    image = np.full((len(bands), *green.shape), 0.2, dtype="float32")
    image[bands.index("B03")], image[bands.index("B11")] = green, swir
    res = segment_scene(image, bands, np.ones_like(water), cfg["baseline"])
    assert iou(res.water, water, res.evaluated) > 0.99
