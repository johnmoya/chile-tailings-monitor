import pytest

from ctm.data.deposits import (
    DepositSchemaError,
    TestSplitLockedError,
    assert_disjoint_splits,
    load_deposits,
)
from ctm.utils.config import resolve_path
from tests.helpers import write_deposits


def test_shipped_registry_is_a_single_todo_placeholder(cfg):
    reg = load_deposits(resolve_path(cfg, "deposits"))
    assert len(reg.ids) == 1
    assert bool(reg.gdf["todo"].iloc[0])
    assert reg.real().ids == []
    with pytest.raises(DepositSchemaError, match="placeholder"):
        load_deposits(resolve_path(cfg, "deposits"), allow_placeholders=False)


def test_valid_registry_and_split_access(tmp_path):
    p = write_deposits(tmp_path / "d.geojson", [
        {"id": "dep-a", "split": "train"}, {"id": "dep-b", "split": "val"},
        {"id": "dep-c", "split": "test"},
    ])
    reg = load_deposits(p, allow_placeholders=False)
    assert reg.ids_for("train") == ["dep-a"]
    assert reg.ids_for("val") == ["dep-b"]
    assert reg.split_of("dep-c") == "test"


def test_test_split_is_locked_by_default(tmp_path):
    p = write_deposits(tmp_path / "d.geojson", [{"id": "dep-c", "split": "test"}])
    reg = load_deposits(p)
    with pytest.raises(TestSplitLockedError):
        reg.ids_for("test")
    assert reg.ids_for("test", allow_test=True) == ["dep-c"]


@pytest.mark.parametrize("rows,match", [
    ([{"id": "a", "split": "train"}, {"id": "a", "split": "val"}], "duplicate"),
    ([{"id": "a", "split": "holdout"}], "split must be"),
    ([{"id": "Bad Id", "split": "train"}], "invalid id"),
    ([{"id": "a", "split": "train", "geometry": None}], "missing geometry"),
    ([{"id": "a", "split": "train",
       "geometry": {"type": "Point", "coordinates": [-70, -30]}}], "Polygon"),
])
def test_schema_errors(tmp_path, rows, match):
    p = write_deposits(tmp_path / "d.geojson", rows)
    with pytest.raises(DepositSchemaError, match=match):
        load_deposits(p)


def test_missing_required_property(tmp_path):
    p = write_deposits(tmp_path / "d.geojson", [{"id": "a"}])
    with pytest.raises(DepositSchemaError, match="split"):
        load_deposits(p)


def test_disjoint_splits_detects_leakage():
    assert_disjoint_splits({"train": ["a", "b"], "val": ["c"], "test": ["d"]})
    with pytest.raises(AssertionError, match="leakage"):
        assert_disjoint_splits({"train": ["a"], "test": ["a"]})
