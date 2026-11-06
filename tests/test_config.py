from pathlib import Path

from ctm.utils.config import load_config, parse_override, resolve_path


def test_default_config_has_required_keys(cfg):
    assert cfg["stac"]["bands"] == ["B02", "B03", "B04", "B08", "B11", "B12"]
    assert cfg["tiling"]["tile_size"] == 256
    assert cfg["model"]["in_channels"] == len(cfg["stac"]["bands"])
    assert cfg["train"]["amp_dtype"] == "bfloat16"
    assert isinstance(cfg["seed"], int)
    for b in cfg["stac"]["bands"] + ["SCL"]:
        assert b in cfg["stac"]["asset_keys"]


def test_paths_resolve_against_repo_root(cfg):
    root = Path(__file__).resolve().parents[1]
    assert resolve_path(cfg, "deposits") == root / "configs" / "deposits.geojson"
    assert resolve_path(cfg, "deposits").is_file()


def test_overrides_parse_yaml_values():
    assert parse_override("train.epochs=3") == ("train.epochs", 3)
    assert parse_override("model.encoder_weights=null") == ("model.encoder_weights", None)
    cfg = load_config(overrides=["train.epochs=3", "train.device=cpu"])
    assert cfg["train"]["epochs"] == 3 and cfg["train"]["device"] == "cpu"


def test_inherit(tmp_path):
    child = tmp_path / "configs" / "child.yaml"
    child.parent.mkdir()
    default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    child.write_text(f"inherit: {default}\ntrain:\n  epochs: 2\n")
    cfg = load_config(child)
    assert cfg["train"]["epochs"] == 2
    assert cfg["train"]["batch_size"] == load_config()["train"]["batch_size"]
