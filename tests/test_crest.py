import json

import pytest

from ctm.data.crest import load_crests
from ctm.data.deposits import DepositSchemaError, load_deposits
from ctm.utils.config import resolve_path
from tests.helpers import write_deposits


def _write(path, feats):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path


def _line(did, coords=((-70.0, -30.0), (-69.99, -30.0)), gtype="LineString"):
    return {"type": "Feature", "properties": {"deposit_id": did},
            "geometry": {"type": gtype, "coordinates": [list(c) for c in coords]}}


def test_shipped_crest_file_is_empty_schema(cfg):
    crests = load_crests(resolve_path(cfg, "crest_lines"))
    assert crests.deposit_ids == []
    doc = json.loads(resolve_path(cfg, "crest_lines").read_text())
    assert "deposit_id" in doc["schema"]["required_properties"]


def test_valid_crests(tmp_path):
    reg = load_deposits(write_deposits(tmp_path / "d.geojson", [{"id": "dep-a", "split": "train"}]))
    c = load_crests(_write(tmp_path / "c.geojson", [_line("dep-a")]), reg)
    assert c.deposit_ids == ["dep-a"]
    assert c.line("dep-a").geom_type == "LineString"
    with pytest.raises(KeyError):
        c.line("nope")


@pytest.mark.parametrize("feats,match", [
    ([_line("dep-a"), _line("dep-a")], "more than one"),
    ([_line("ghost")], "unknown deposit_id"),
    ([{"type": "Feature", "properties": {"deposit_id": "dep-a"},
       "geometry": {"type": "Point", "coordinates": [-70, -30]}}], "LineString"),
])
def test_invalid_crests(tmp_path, feats, match):
    reg = load_deposits(write_deposits(tmp_path / "d.geojson", [{"id": "dep-a", "split": "train"}]))
    with pytest.raises(DepositSchemaError, match=match):
        load_crests(_write(tmp_path / "c.geojson", feats), reg)
