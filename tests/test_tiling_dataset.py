import numpy as np
import pytest

from ctm.data.dataset import SyntheticWaterDataset, TileDataset
from ctm.data.deposits import TestSplitLockedError, load_deposits
from ctm.data.tiling import make_tiles, save_tile, tile_windows
from tests.helpers import write_deposits


def test_tile_windows_cover_extent():
    wins = list(tile_windows(600, 520, 256, 256))
    rows = sorted({r for r, _ in wins})
    cols = sorted({c for _, c in wins})
    assert rows == [0, 256, 344] and cols == [0, 256, 264]


def test_make_tiles_pads_and_filters():
    img = np.random.rand(6, 300, 200).astype("float32")
    lab = np.zeros((300, 200), np.uint8)
    lab[:, 150:] = 255
    tiles = make_tiles(img, lab, "dep-a", "s1", 256, 256, 255, min_valid_fraction=0.3)
    assert tiles and all(t.image.shape == (6, 256, 256) for t in tiles)
    assert all(t.label.shape == (256, 256) for t in tiles)
    none = make_tiles(img, lab, "dep-a", "s1", 256, 256, 255, min_valid_fraction=0.99)
    assert none == []


def test_tile_dataset_selects_by_deposit_and_locks_test(tmp_path):
    reg = load_deposits(write_deposits(tmp_path / "d.geojson", [
        {"id": "dep-a", "split": "train"}, {"id": "dep-b", "split": "val"},
        {"id": "dep-c", "split": "test"},
    ]))
    tiles_dir = tmp_path / "tiles"
    img = np.random.rand(6, 256, 256).astype("float32")
    lab = np.zeros((256, 256), np.uint8)
    for did in ("dep-a", "dep-b", "dep-c"):
        for t in make_tiles(img, lab, did, "s1", 256, 256, 255, 0.0):
            save_tile(t, tiles_dir)
    tr = TileDataset.for_split(tiles_dir, reg, "train")
    va = TileDataset.for_split(tiles_dir, reg, "val")
    assert len(tr) == 1 and tr[0]["deposit_id"] == "dep-a"
    assert len(va) == 1 and va[0]["deposit_id"] == "dep-b"
    assert tr[0]["image"].shape == (6, 256, 256)
    with pytest.raises(TestSplitLockedError):
        TileDataset.for_split(tiles_dir, reg, "test")


def test_synthetic_dataset_is_deterministic():
    a = SyntheticWaterDataset(2, size=64, seed=3)[1]
    b = SyntheticWaterDataset(2, size=64, seed=3)[1]
    assert (a["image"] == b["image"]).all() and (a["label"] == b["label"]).all()
    assert set(a["label"].unique().tolist()) <= {0, 1, 255}
