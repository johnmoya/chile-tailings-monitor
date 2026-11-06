import math

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box

from ctm.metrics.derived import (
    beach_fraction,
    inset_mask,
    pond_area_m2,
    pond_crest_distance_m,
)

# 10 m grid, origin (0, 1000): pixel (r, c) covers x in [10c, 10c+10], y in [990-10r, 1000-10r]
T = Affine(10, 0, 0, 0, -10, 1000)
SHAPE = (100, 100)


def test_pond_area():
    w = np.zeros(SHAPE, bool)
    w[10:20, 10:30] = True
    assert pond_area_m2(w, T) == 200 * 100.0
    valid = np.ones(SHAPE, bool)
    valid[10:15] = False
    assert pond_area_m2(w, T, valid) == 100 * 100.0


def test_inset_shrinks_polygon_by_given_distance():
    poly = box(100, 100, 900, 900)               # 800 m square -> 80 x 80 px
    full = inset_mask(poly, 0, T, SHAPE)
    inner = inset_mask(poly, 30, T, SHAPE)        # 740 m square -> 74 x 74 px
    assert full.sum() == 80 * 80
    assert inner.sum() == 74 * 74
    assert not inset_mask(poly, 500, T, SHAPE).any()


def test_beach_fraction_inset_minus_pond():
    poly = box(100, 100, 900, 900)
    inset = inset_mask(poly, 30, T, SHAPE)        # 74 x 74 = 5476 px
    water = np.zeros(SHAPE, bool)
    water[40:60, 40:60] = True                    # 400 px, fully inside the inset
    assert beach_fraction(water, inset) == pytest.approx(1 - 400 / 5476)
    # Water hugging the polygon edge (outside the inset) does not reduce the beach.
    edge = water.copy()
    edge[10:13, 10:90] = True
    assert beach_fraction(edge, inset) == pytest.approx(1 - 400 / 5476)


def test_beach_fraction_respects_validity():
    poly = box(100, 100, 900, 900)
    inset = inset_mask(poly, 30, T, SHAPE)
    water = np.zeros(SHAPE, bool)
    valid = np.ones(SHAPE, bool)
    valid[:, :50] = False                         # half the inset unknown -> NaN at 0.8
    assert math.isnan(beach_fraction(water, inset, valid, min_valid_fraction=0.8))
    assert beach_fraction(water, inset, valid, min_valid_fraction=0.4) == 1.0
    assert math.isnan(beach_fraction(water, np.zeros(SHAPE, bool)))


def test_pond_crest_distance_in_metres():
    water = np.zeros(SHAPE, bool)
    water[40:60, 10:20] = True                    # pond x in [100, 200] m
    crest = LineString([(500, 0), (500, 1000)])   # vertical crest at x = 500 m
    assert pond_crest_distance_m(water, T, crest) == pytest.approx(300.0)
    diag = LineString([(300, 1000), (300 + 1, 1000 - 1000)])   # ~ vertical at x~300
    assert pond_crest_distance_m(water, T, diag) == pytest.approx(100.0, abs=1.0)


def test_pond_crest_distance_edges_and_touching():
    water = np.zeros(SHAPE, bool)
    water[0:10, 0:10] = True                      # pond corner at (100, 900)
    crest = LineString([(130, 860), (200, 860)])  # 30 m right, 40 m below -> 50 m
    assert pond_crest_distance_m(water, T, crest) == pytest.approx(50.0)
    touching = LineString([(50, 950), (500, 950)])
    assert pond_crest_distance_m(water, T, touching) == 0.0
    assert math.isnan(pond_crest_distance_m(np.zeros(SHAPE, bool), T, crest))


def test_two_ponds_take_the_nearest():
    water = np.zeros(SHAPE, bool)
    water[50:55, 0:5] = True                      # x in [0, 50]
    water[50:55, 60:65] = True                    # x in [600, 650]
    crest = LineString([(800, 0), (800, 1000)])
    assert pond_crest_distance_m(water, T, crest) == pytest.approx(150.0)
