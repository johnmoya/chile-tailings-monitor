"""PyTorch datasets. Tiles are selected by DEPOSIT id, never by tile."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ctm.data.deposits import DepositRegistry


class TileDataset(Dataset):
    """Reads ``<tiles_dir>/<deposit_id>/*.npz`` for the given deposits only."""

    def __init__(self, tiles_dir: str | Path, deposit_ids: Sequence[str]):
        self.tiles_dir = Path(tiles_dir)
        self.deposit_ids = list(deposit_ids)
        self.files: list[Path] = []
        for did in self.deposit_ids:
            self.files.extend(sorted((self.tiles_dir / did).glob("*.npz")))

    @classmethod
    def for_split(cls, tiles_dir: str | Path, registry: DepositRegistry, split: str) -> TileDataset:
        # ids_for("test") raises unless allow_test=True: training code cannot reach test tiles.
        return cls(tiles_dir, registry.ids_for(split))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as z:
            image = np.nan_to_num(z["image"].astype("float32"), nan=0.0)
            label = z["label"].astype("int64")
            meta = json.loads(str(z["meta"]))
        if meta["deposit_id"] not in self.deposit_ids:
            raise RuntimeError(f"tile {self.files[idx]} belongs to {meta['deposit_id']}")
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label),
                "deposit_id": meta["deposit_id"]}


class SyntheticWaterDataset(Dataset):
    """Random 6-band scenes with elliptical 'ponds' — for smoke tests only.

    Water pixels: high green, low SWIR (MNDWI > 0). Land: the opposite.
    Some pixels are set to ``ignore_index`` to exercise the masked loss.
    """

    def __init__(self, n: int, size: int = 256, channels: int = 6, seed: int = 0,
                 ignore_index: int = 255, green_idx: int = 1, swir_idx: int = 4):
        self.n, self.size, self.channels = n, size, channels
        self.seed, self.ignore_index = seed, ignore_index
        self.green_idx, self.swir_idx = green_idx, swir_idx

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        cy, cx = rng.uniform(0.3, 0.7, 2) * s
        ry, rx = rng.uniform(0.1, 0.3, 2) * s
        water = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        image = rng.normal(0.15, 0.02, (self.channels, s, s)).astype("float32")
        image[self.green_idx][water] = rng.normal(0.10, 0.01, water.sum())
        image[self.swir_idx][water] = rng.normal(0.02, 0.005, water.sum())
        image[self.green_idx][~water] = rng.normal(0.08, 0.01, (~water).sum())
        image[self.swir_idx][~water] = rng.normal(0.25, 0.03, (~water).sum())
        label = water.astype("int64")
        label[: s // 16, :] = self.ignore_index  # a strip of "cloud"
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label),
                "deposit_id": f"synthetic-{idx % 4}"}
