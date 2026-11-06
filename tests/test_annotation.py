import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box

from ctm.data.stac import Scene
from ctm.labels.annotation import export_scene, stretch_uint8, validate_scene_dir

import validate_annotations

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _scene(h=80, w=90):
    rng = np.random.default_rng(0)
    image = rng.uniform(0.02, 0.3, (6, h, w)).astype("float32")
    image[:, 0, 0] = np.nan
    t = Affine(10, 0, 300000, 0, -10, 6700000)
    poly = box(300100, 6699300, 300800, 6699900)
    from ctm.utils.geo import polygon_mask

    pm = polygon_mask(poly, t, (h, w))
    sc = Scene(image=image, valid=np.ones((h, w), bool), scl=np.full((h, w), 5, np.uint8),
               polygon_mask=pm, bands=BANDS, transform=t, crs="EPSG:32719",
               item_id="S2A_19JBG_20260715_0_L2A", datetime="2026-07-15T14:40:00+00:00",
               meta={"tile": "19JBG", "processing_baseline": "05.11",
                     "scene_cloud_cover": 3.2})
    return sc, poly


@pytest.fixture
def kit(tmp_path, cfg):
    sc, poly = _scene()
    d = export_scene(sc, "dep-v", poly, tmp_path / "annotation", cfg, {"scene_id": "x"})
    return tmp_path / "annotation", d, sc


def test_export_layout_and_no_forbidden_products(kit):
    root, d, sc = kit
    assert d == root / "dep-v" / "2026-07-15"
    names = {p.name for p in d.iterdir()}
    assert names == {"rgb.tif", "falsecolor.tif", "mask.tif", "polygon.geojson", "meta.json"}
    for n in names:
        assert not any(f in n.lower() for f in ("mndwi", "baseline", "otsu", "scl"))
    meta = json.loads((d / "meta.json").read_text())
    assert not any("mndwi" in k.lower() or "baseline" == k for k in meta)


def test_export_grids_and_template(kit):
    _, d, sc = kit
    with rasterio.open(d / "rgb.tif") as r, rasterio.open(d / "falsecolor.tif") as f, \
            rasterio.open(d / "mask.tif") as m:
        assert r.count == 3 and f.count == 3 and m.count == 1
        assert r.dtypes[0] == f.dtypes[0] == m.dtypes[0] == "uint8"
        for ds in (f, m):
            assert (ds.crs, ds.transform, ds.shape) == (r.crs, r.transform, r.shape)
        assert r.transform == sc.transform and r.shape == sc.polygon_mask.shape
        assert r.descriptions == ("B04", "B03", "B02")
        assert f.descriptions == ("B12", "B08", "B04")
        assert (m.read(1) == 255).all()
        assert r.read(1)[0, 0] == 0   # no-data pixel kept as 0


def test_export_does_not_overwrite_existing_mask(kit, cfg):
    root, d, _ = kit
    with rasterio.open(d / "mask.tif", "r+") as m:
        a = m.read(1)
        a[:] = 0
        m.write(a, 1)
    sc, poly = _scene()
    export_scene(sc, "dep-v", poly, root, cfg)
    with rasterio.open(d / "mask.tif") as m:
        assert (m.read(1) == 0).all()


def test_stretch_joint_vs_per_band():
    stack = np.stack([np.linspace(0, 1, 100).reshape(10, 10),
                      np.linspace(0, 2, 100).reshape(10, 10)]).astype("float32")
    region = np.ones((10, 10), bool)
    joint = stretch_uint8(stack, region, 0, 100, True)
    per = stretch_uint8(stack, region, 0, 100, False)
    assert per[0].max() == per[1].max() == 255
    assert joint[0].max() < 255 and joint[1].max() == 255
    assert joint.min() >= 1


def _paint(d, fn):
    with rasterio.open(d / "mask.tif", "r+") as m:
        a = m.read(1)
        m.write(fn(a), 1)


def test_validator_statuses(kit, cfg):
    root, d, _ = kit
    min_cov = cfg["annotation"]["min_coverage"]
    assert validate_scene_dir(d, min_cov).status == "pending"

    def partial(a):
        a[:40] = 0
        return a
    _paint(d, partial)
    rep = validate_scene_dir(d, min_cov)
    assert rep.status == "incomplete" and 0 < rep.coverage < min_cov

    def full(a):
        a[:] = 0
        a[30:50, 30:50] = 1
        return a
    _paint(d, full)
    rep = validate_scene_dir(d, min_cov)
    assert rep.status == "ok" and rep.coverage == 1.0 and 0 < rep.water_fraction < 1

    def bad(a):
        a[0, 0] = 7
        return a
    _paint(d, bad)
    rep = validate_scene_dir(d, min_cov)
    assert rep.status == "error" and "invalid mask values" in rep.problems[0]


def test_validator_detects_grid_mismatch_and_forbidden_files(kit, cfg):
    root, d, _ = kit
    with rasterio.open(d / "mask.tif") as m:
        prof = m.profile
    prof.update(transform=Affine(10, 0, 300005, 0, -10, 6700000))
    with rasterio.open(d / "mask.tif", "w", **prof) as m:
        m.write(np.full((1, prof["height"], prof["width"]), 0, np.uint8))
    (d / "mndwi.tif").write_bytes(b"")
    rep = validate_scene_dir(d, cfg["annotation"]["min_coverage"])
    assert rep.status == "error"
    assert any("grid differs" in p for p in rep.problems)
    assert any("forbidden" in p for p in rep.problems)


def test_validate_annotations_script_exit_codes(kit):
    root, d, _ = kit
    assert validate_annotations.main(["--root", str(root)]) == 0
    assert validate_annotations.main(["--root", str(root), "--require-complete"]) == 1
    _paint(d, lambda a: np.where(a == 255, 0, a).astype("uint8"))
    assert validate_annotations.main(["--root", str(root), "--require-complete"]) == 0
