import json

import pytest
import torch

from ctm.data.dataset import SyntheticWaterDataset
from ctm.models.unet import build_model
from ctm.train.loop import load_checkpoint, masked_bce_dice, train


def _cfg(tmp_cfg, device, size, batch, amp):
    cfg = dict(tmp_cfg)
    cfg["train"] = dict(cfg["train"], device=device, epochs=1, batch_size=batch,
                        num_workers=0, amp=amp)
    cfg["model"] = dict(cfg["model"], encoder_weights=None)
    cfg["tiling"] = dict(cfg["tiling"], tile_size=size)
    return cfg


def test_model_shapes(cfg):
    m = build_model(dict(cfg["model"], encoder_weights=None))
    out = m(torch.zeros(2, 6, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_masked_loss_ignores_pixels():
    logits = torch.zeros(1, 1, 4, 4)
    t1 = torch.full((1, 4, 4), 255)
    t1[0, 0, 0] = 1
    t2 = t1.clone()
    t2[0, 3, 3] = 255
    assert masked_bce_dice(logits, t1, 255) == masked_bce_dice(logits, t2, 255)
    assert masked_bce_dice(logits, torch.full((1, 4, 4), 255), 255) == 0


def test_one_epoch_cpu_writes_checkpoints_and_log(tmp_cfg, tmp_path):
    cfg = _cfg(tmp_cfg, "cpu", 64, 4, amp=False)
    ds = SyntheticWaterDataset(8, size=64, seed=0)
    res = train(cfg, ds, SyntheticWaterDataset(4, size=64, seed=1), tmp_path / "run")
    assert (tmp_path / "run" / "last.pt").is_file()
    assert (tmp_path / "run" / "best.pt").is_file()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["epoch"] == 1 and {"iou", "f1", "loss"} <= set(rec["val"])
    model, state = load_checkpoint(res.last_checkpoint)
    assert state["epoch"] == 1


def test_seeded_runs_are_reproducible_cpu(tmp_cfg, tmp_path):
    cfg = _cfg(tmp_cfg, "cpu", 32, 4, amp=False)
    ds = SyntheticWaterDataset(4, size=32, seed=0)
    a = train(cfg, ds, None, tmp_path / "a").history[0]["train"]["loss"]
    b = train(cfg, ds, None, tmp_path / "b").history[0]["train"]["loss"]
    assert a == pytest.approx(b, rel=1e-6)


@pytest.mark.gpu
def test_one_epoch_gpu_bf16(tmp_cfg, tmp_path):
    cfg = _cfg(tmp_cfg, "cuda", 256, tmp_cfg["train"]["batch_size"], amp=True)
    ds = SyntheticWaterDataset(64, size=256, seed=0)
    res = train(cfg, ds, SyntheticWaterDataset(16, size=256, seed=1), tmp_path / "gpu")
    assert res.peak_vram_mib is not None and res.peak_vram_mib > 0
    assert res.peak_vram_mib < 11 * 1024  # fits on a 12 GB card
    rec = res.history[0]
    assert rec["train"]["loss"] == rec["train"]["loss"]  # not NaN
    print(f"\npeak VRAM: {res.peak_vram_mib:.0f} MiB; val IoU {rec['val']['iou']:.3f}")
