import datetime

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
    s = cfg["stac"]
    # Shadow (3) is kept unless the shadow rule fires.
    valid = scl_valid_mask(scl, s["scl_invalid_classes"], s["scl_shadow_class"], False)
    assert valid.ravel().tolist() == [
        False, False, True, True, True, True, True, True, False, False, False, False]
    valid = scl_valid_mask(scl, s["scl_invalid_classes"], s["scl_shadow_class"], True)
    assert not valid.ravel()[3]


def test_item_tile_from_properties_and_id():
    from pystac import Item

    from ctm.data.stac import item_tile

    def mk(props, iid="x"):
        return Item(iid, None, None, datetime.datetime(2020, 1, 1), props)

    assert item_tile(mk({"grid:code": "MGRS-19JBG"})) == "19JBG"
    assert item_tile(mk({"mgrs:utm_zone": 19, "mgrs:latitude_band": "J",
                         "mgrs:grid_square": "BG"})) == "19JBG"
    assert item_tile(mk({}, "S2A_19JCH_20190129_0_L2A")) == "19JCH"


# --- FIX-1: shadow rule ------------------------------------------------------------

def _shadow_setup():
    """20 m SCL window: inner bbox = 1 km square, extended by 1.5 km on each side.

    Inner bbox spans x,y in [1500, 2500] m of a 4 km x 4 km window (200 x 200 px).
    A 'lagoon' inside the bbox is labelled SCL=3 (dark water misread as shadow);
    the rest is bare soil (5).
    """
    from affine import Affine

    t = Affine(20, 0, 0, 0, -20, 4000)       # origin top-left at (0, 4000)
    ext = np.full((200, 200), 5, np.uint8)
    bounds = (1500.0, 1500.0, 2500.0, 2500.0)
    # inner rows/cols for y,x in [1500,2500]: cols 75..124, rows 75..124
    ext[90:110, 90:110] = 3                  # the "lagoon"
    inner = ext[75:125, 75:125].copy()
    return ext, inner, t, bounds


def test_shadow_without_clouds_is_kept(cfg):
    from ctm.data.stac import scl_validity

    ext, inner, t, bounds = _shadow_setup()
    valid, shadow = scl_validity(inner, ext, t, bounds, cfg["stac"])
    assert shadow is False
    assert valid[15:35, 15:35].all()          # lagoon pixels survive


def test_shadow_with_cloud_within_radius_is_masked(cfg):
    from ctm.data.stac import scl_validity

    ext, inner, t, bounds = _shadow_setup()
    ext[60, 100] = 9   # cloud ~ 290 m north of the bbox edge (row 60 -> y = 2790)
    valid, shadow = scl_validity(inner, ext, t, bounds, cfg["stac"])
    assert shadow is True
    assert not valid[15:35, 15:35].any()


def test_cloud_beyond_radius_does_not_trigger(cfg):
    from ctm.data.stac import clouds_near_bbox

    ext, _, t, bounds = _shadow_setup()
    ext[0, 0] = 8      # corner pixel centre (10, 3990): ~2.1 km from the bbox corner
    assert not clouds_near_bbox(ext, t, bounds, 1500, [8, 9, 10])
    assert clouds_near_bbox(ext, t, bounds, 2200, [8, 9, 10])
    ext[0, 0] = 11     # snow is not a shadow-casting class
    assert not clouds_near_bbox(ext, t, bounds, 5000, [8, 9, 10])


def test_scl_fractions():
    from ctm.data.stac import scl_fractions

    scl = np.array([[3, 3], [6, 9]], np.uint8)
    f = scl_fractions(scl, np.ones((2, 2), bool))
    assert f[3] == 0.5 and f[6] == 0.25 and f[9] == 0.25 and f[0] == 0


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
    assert scene.scl.shape == (h, w) and scene.scl.dtype == np.uint8
    assert isinstance(scene.meta["shadow_masked"], bool) and scene.meta["tile"]


@pytest.mark.network
def test_catalog_and_annotation_export_end_to_end(cfg, tmp_path):
    """Catalog two months for the smoke box, export one scene, validate the kit."""
    from functools import partial

    from ctm.data.catalog import catalog_deposit
    from ctm.data.ingest import load_item
    from ctm.data.stac import item_crs, load_scene
    from ctm.labels.annotation import export_scene, validate_scene_dir
    from ctm.utils.geo import reproject_geometry

    search = partial(search_items_window, start="2019-01-01", end="2019-02-28")
    df, counts = catalog_deposit("smoke-box", "val", SMOKE_BOX, cfg, search_fn=search,
                                 workers=4)
    assert len(df) >= 3 and df["tile"].nunique() == 1
    assert counts[df["tile"].iloc[0]] == len(df)
    assert df["valid_fraction"].between(0, 1).all()
    assert np.allclose(df[[f"scl_{k}" for k in range(12)]].sum(axis=1), 1.0)

    row = df.sort_values("valid_fraction").iloc[-1]
    item = load_item(row["item_href"])
    scene = load_scene(item, SMOKE_BOX, cfg["stac"])
    poly = reproject_geometry(SMOKE_BOX, "EPSG:4326", item_crs(item))
    d = export_scene(scene, "smoke-box", poly, tmp_path / "annotation", cfg)
    rep = validate_scene_dir(d, cfg["annotation"]["min_coverage"])
    assert rep.status == "pending" and not rep.problems


def search_items_window(geom, stac_cfg, start, end, **kw):
    from ctm.data.stac import search_items

    return search_items(geom, stac_cfg, start, end, **kw)
