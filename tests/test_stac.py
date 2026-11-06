import numpy as np
import pytest
from shapely.geometry import box

from ctm.data.stac import _acquisition_key, scl_valid_mask


def test_acquisition_key_handles_reprocessed_ids():
    assert _acquisition_key("S2A_19JBG_20190129_1_L2A") == ("S2A_19JBG_20190129", 1)
    assert _acquisition_key("S2B_T19JBG_20240131T145617_L2A")[1] == 0


def test_reproject_geometry_roundtrip_and_polygon_mask():
    from affine import Affine

    from ctm.utils.geo import polygon_mask, reproject_geometry

    g = box(-71.0, -30.0, -70.99, -29.99)
    utm = reproject_geometry(g, "EPSG:4326", "EPSG:32719")
    assert 290_000 < utm.centroid.x < 310_000 and 6_670_000 < utm.centroid.y < 6_690_000
    back = reproject_geometry(utm, "EPSG:32719", "EPSG:4326")
    assert back.symmetric_difference(g).area < 1e-10
    minx, _, _, maxy = utm.bounds
    mask = polygon_mask(utm, Affine(10, 0, minx, 0, -10, maxy), (130, 130))
    assert 0.5 < mask.mean() < 0.75   # ~1.1 km x 0.97 km square inside a 1.3 km box


def test_scl_mask_default_classes(cfg):
    scl = np.arange(12, dtype=np.uint8).reshape(3, 4)
    valid = scl_valid_mask(scl, cfg["stac"]["scl_invalid_classes"])
    assert valid.ravel().tolist() == [
        False, False, True, False, True, True, True, True, False, False, False, False]


# Arbitrary ~1 km box near La Serena used only to exercise the STAC/COG path.
# It is NOT a tailings deposit.
SMOKE_BOX = box(-71.255, -29.955, -71.245, -29.945)


@pytest.mark.network
def test_stac_single_small_scene(cfg):
    from ctm.data.stac import load_scene, search_items

    items = search_items(SMOKE_BOX, cfg["stac"], "2019-01-01", "2019-02-28", max_items=10)
    assert items, "no Sentinel-2 L2A items found for the smoke box"
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    scene = load_scene(item, SMOKE_BOX, cfg["stac"])

    c, h, w = scene.image.shape
    assert c == len(cfg["stac"]["bands"])
    # ~1 km + 2 x 300 m buffer at 10 m -> ~160 px per side (+/- grid alignment)
    assert 140 <= h <= 180 and 140 <= w <= 180
    assert scene.polygon_mask.shape == (h, w) and scene.polygon_mask.any()
    assert scene.valid.shape == (h, w)
    finite = scene.image[:, scene.valid]
    assert finite.size and np.isfinite(finite).all()
    # Surface reflectance after scale/offset: physically plausible range.
    assert np.nanpercentile(finite, 1) > -0.2 and np.nanpercentile(finite, 99) < 1.5
    assert scene.crs.startswith("EPSG:327")
