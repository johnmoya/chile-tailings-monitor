import datetime

import numpy as np
import pandas as pd
import pytest
from pystac import Item
from shapely.geometry import box, mapping

from ctm.data.catalog import SCL_COLS, build_catalog, choose_tile
from ctm.data.deposits import load_deposits
from ctm.data.sampling import annotation_selection, season_of, series_scenes, train_sample
from ctm.data.stac import SclSummary
from tests.helpers import fake_catalog, write_deposits


def _item(tile: str, day: int) -> Item:
    dt = datetime.datetime(2020, 1, 1) + datetime.timedelta(days=day)
    it = Item(f"S2A_{tile}_{dt:%Y%m%d}_0_L2A", mapping(box(-72, -31, -70, -29)),
              [-72, -31, -70, -29], dt, {"grid:code": f"MGRS-{tile}", "eo:cloud_cover": 5.0,
                                         "s2:processing_baseline": "02.14"})
    it.set_self_href(f"https://example.invalid/{it.id}.json")
    return it


def test_choose_tile_most_scenes_then_alphabetical():
    items = [_item("19JBG", d) for d in range(5)] + [_item("19JCG", d) for d in range(7)]
    assert choose_tile(items)[0] == "19JCG"
    tied = [_item("19JCG", d) for d in range(3)] + [_item("19JBG", d) for d in range(3)]
    tile, counts = choose_tile(tied)
    assert tile == "19JBG" and counts == {"19JCG": 3, "19JBG": 3}
    with pytest.raises(ValueError):
        choose_tile([])


def test_build_catalog_fixed_tile_and_columns(tmp_path, cfg):
    reg = load_deposits(write_deposits(tmp_path / "d.geojson", [
        {"id": "dep-a", "split": "train"}, {"id": "dep-b", "split": "test"}]))
    items = [_item("19JBG", d) for d in range(4)] + [_item("19JCG", d) for d in range(6)]

    def search(geom, stac_cfg, **kw):
        assert kw == {"max_cloud": None, "require_full_coverage": True}
        return items

    def summarize(item, geom, stac_cfg):
        f = {k: 0.0 for k in range(12)}
        f[5] = 1.0
        return SclSummary(0.97, f, False, 1234)

    cat, tiles = build_catalog(reg, cfg, tmp_path / "parts", search, summarize)
    assert set(cat["tile"]) == {"19JCG"}
    assert len(cat) == 12 and set(cat["deposit_id"]) == {"dep-a", "dep-b"}
    required = {"deposit_id", "scene_id", "date", "tile", "processing_baseline",
                "valid_fraction", "scene_cloud_cover"}
    assert required <= set(cat.columns) and set(SCL_COLS) <= set(cat.columns)
    assert tiles.loc[tiles["chosen"], "tile"].tolist() == ["19JCG", "19JCG"]
    # cached parts are reused (search would now fail)
    cat2, _ = build_catalog(reg, cfg, tmp_path / "parts", None, None)
    assert len(cat2) == 12


def test_train_sample_two_per_month_best_first():
    cat = fake_catalog(["dep-a"], "2020-01-01", "2020-12-31", every=3, seed=1, split="train")
    s = train_sample(cat, ["dep-a"], per_month=2, min_valid=0.9, min_days_apart=10)
    per_month = s.groupby(s["date"].dt.to_period("M")).size()
    assert per_month.max() <= 2
    for _, g in s.groupby(s["date"].dt.to_period("M")):
        if len(g) == 2:
            assert abs((g["date"].iloc[1] - g["date"].iloc[0]).days) >= 10
    assert (s["valid_fraction"] >= 0.9).all()
    assert train_sample(cat, ["other"], 2, 0.9, 10).empty


def test_series_threshold():
    cat = fake_catalog(["dep-a"], seed=2)
    s = series_scenes(cat, 0.8)
    assert len(s) == int((cat["valid_fraction"] >= 0.8).sum())


def test_season_of_southern_hemisphere(cfg):
    seasons = cfg["annotation"]["seasons"]
    assert season_of(1, seasons) == "summer" and season_of(7, seasons) == "winter"


def _ann(cfg, cat, did="dep-v", seed=None):
    a = cfg["annotation"]
    return annotation_selection(cat[cat["deposit_id"] == did], a["scenes_per_deposit"],
                                a["seed"] if seed is None else seed, did, a["seasons"],
                                a["min_valid_fraction"], a["mandatory"],
                                a["mandatory_min_valid_fraction"])


def test_annotation_selection_stratified_and_mandatory(cfg):
    cat = fake_catalog(["dep-v"], seed=3)
    sel = _ann(cfg, cat)
    assert len(sel) == 8
    assert sel["season"].value_counts().to_dict() == {
        "summer": 2, "autumn": 2, "winter": 2, "spring": 2}
    months_2026 = set(sel.loc[sel["date"].dt.year == 2026, "date"].dt.month)
    assert {7, 8} <= months_2026          # mandatory Jul + Aug 2026
    assert sel["selection_reason"].str.startswith("mandatory").sum() == 2
    non_mand = sel[~sel["selection_reason"].str.startswith("mandatory")]
    assert non_mand["year"].nunique() == len(non_mand)   # 6 slots -> 6 distinct years
    assert sel["scene_id"].is_unique


def test_annotation_selection_is_deterministic_and_seed_sensitive(cfg):
    cat = fake_catalog(["dep-v"], seed=4)
    a, b = _ann(cfg, cat), _ann(cfg, cat)
    assert a["scene_id"].tolist() == b["scene_id"].tolist()
    c = _ann(cfg, cat, seed=1)
    assert a["scene_id"].tolist() != c["scene_id"].tolist()


def test_annotation_selection_without_2026_winter(cfg):
    cat = fake_catalog(["dep-v"], end="2026-05-31", seed=5)
    sel = _ann(cfg, cat)
    assert len(sel) == 8 and not sel["selection_reason"].str.startswith("mandatory").any()


def test_annotation_selection_short_season_is_filled(cfg):
    cat = fake_catalog(["dep-v"], seed=6)
    cat.loc[cat["date"].dt.month.isin([12, 1, 2]), "valid_fraction"] = 0.5   # cloudy summers
    sel = _ann(cfg, cat)
    assert len(sel) == 8 and (sel["season"] != "summer").all()
    assert np.all(sel["valid_fraction"] >= 0.8)


def test_catalog_dates_are_timestamps():
    cat = fake_catalog(["x"], "2020-01-01", "2020-01-10")
    assert pd.api.types.is_datetime64_any_dtype(cat["date"])
