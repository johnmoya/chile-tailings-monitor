import json

import pytest
import torch
import yaml

from ctm.models.unet import build_model
from ctm.utils.oos_lock import CONFIRM_FLAG
from tests.helpers import synthetic_eval_scene, write_deposits, write_eval_scene

import evaluate_test


@pytest.fixture
def setup(tmp_path, tmp_cfg):
    """A complete, tiny test setup: registry, scenes with references, checkpoint, config."""
    write_deposits(tmp_path / "deposits.geojson", [
        {"id": "dep-a", "split": "train"}, {"id": "dep-t", "split": "test"}])
    bands = tmp_cfg["stac"]["bands"]
    write_eval_scene(tmp_path / "scenes", synthetic_eval_scene(bands, "dep-t", "s1"))
    model_cfg = dict(tmp_cfg["model"], encoder_weights=None)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": build_model(model_cfg).state_dict(), "model_cfg": model_cfg,
                "epoch": 0}, ckpt)
    cfg = {k: v for k, v in tmp_cfg.items() if not k.startswith("_")}
    cfg["train"] = dict(cfg["train"], device="cpu")
    cfg_path = tmp_path / "configs" / "test.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": str(cfg_path), "ckpt": str(ckpt),
            "lock": tmp_path / "reports" / "locks" / "test.lock",
            "out": tmp_path / "reports" / "res.json"}


def _argv(s, confirm=True):
    a = ["--config", s["cfg"], "--checkpoint", s["ckpt"], "--out", str(s["out"])]
    return a + ([CONFIRM_FLAG] if confirm else [])


def test_refuses_without_flag(setup):
    assert evaluate_test.main(_argv(setup, confirm=False)) == 2
    assert not setup["lock"].exists()


def test_preconditions_fail_without_taking_lock(setup, tmp_path):
    argv = _argv(setup)
    argv[argv.index("--checkpoint") + 1] = str(tmp_path / "missing.pt")
    assert evaluate_test.main(argv) == 4
    assert not setup["lock"].exists()


def test_placeholder_registry_blocks_run(setup, tmp_path):
    write_deposits(tmp_path / "deposits.geojson", [
        {"id": "dep-t", "split": "test"}, {"id": "todo-x", "split": "train", "todo": True}])
    assert evaluate_test.main(_argv(setup)) == 4
    assert not setup["lock"].exists()


def test_single_run_then_locked(setup):
    assert evaluate_test.main(_argv(setup)) == 0
    lock = json.loads(setup["lock"].read_text())
    assert lock["status"] == "completed" and lock["test_deposits"] == ["dep-t"]
    res = json.loads(setup["out"].read_text())
    assert {"unet", "baseline"} <= set(res["overall"])
    base = res["overall"]["baseline"]
    assert base["iou_macro_deposit"] > 0.99   # synthetic pond is trivially separable
    assert base["iou_micro_pixel"] > 0.99      # both metrics always reported
    # Second run: refused, results untouched.
    before = setup["out"].read_text()
    assert evaluate_test.main(_argv(setup)) == 3
    assert setup["out"].read_text() == before
