"""Cut scenes into fixed-size tiles and store them as .npz, grouped by deposit."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Tile:
    image: np.ndarray        # float32 (C, T, T), NaN -> 0 is done at load time
    label: np.ndarray        # uint8 (T, T) in {0, 1, ignore_index}
    deposit_id: str
    scene_id: str
    row: int
    col: int


def tile_windows(height: int, width: int, size: int, stride: int) -> Iterator[tuple[int, int]]:
    """Top-left corners covering the full extent; the last row/col is snapped to the edge."""
    if height < size or width < size:
        yield 0, 0
        return
    rows = list(range(0, height - size + 1, stride))
    cols = list(range(0, width - size + 1, stride))
    if rows[-1] != height - size:
        rows.append(height - size)
    if cols[-1] != width - size:
        cols.append(width - size)
    for r in rows:
        for c in cols:
            yield r, c


def pad_to(arr: np.ndarray, size: int, value: float) -> np.ndarray:
    h, w = arr.shape[-2:]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(arr, pad, constant_values=value)


def make_tiles(
    image: np.ndarray,
    label: np.ndarray,
    deposit_id: str,
    scene_id: str,
    size: int,
    stride: int,
    ignore_index: int,
    min_valid_fraction: float,
) -> list[Tile]:
    image = pad_to(image, size, np.nan)
    label = pad_to(label, size, ignore_index)
    tiles = []
    for r, c in tile_windows(*label.shape, size, stride):
        lab = label[r : r + size, c : c + size]
        if (lab != ignore_index).mean() < min_valid_fraction:
            continue
        img = image[:, r : r + size, c : c + size]
        tiles.append(Tile(img.astype("float32"), lab.astype("uint8"), deposit_id, scene_id, r, c))
    return tiles


def save_tile(tile: Tile, root: Path) -> Path:
    out = root / tile.deposit_id / f"{tile.scene_id}_r{tile.row:05d}_c{tile.col:05d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"deposit_id": tile.deposit_id, "scene_id": tile.scene_id,
            "row": tile.row, "col": tile.col}
    np.savez_compressed(out, image=tile.image, label=tile.label, meta=json.dumps(meta))
    return out
