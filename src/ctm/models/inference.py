"""Sliding-window inference over a full scene with overlap blending."""

from __future__ import annotations

import numpy as np
import torch

from ctm.data.tiling import pad_to, tile_windows


@torch.no_grad()
def predict_scene(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device | str = "cpu",
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Return water probability (H, W) float32 for a (C, H, W) image (NaN -> 0)."""
    model.eval()
    c, h, w = image.shape
    img = pad_to(np.nan_to_num(image, nan=0.0).astype("float32"), tile_size, 0.0)
    ph, pw = img.shape[-2:]
    prob = np.zeros((ph, pw), dtype="float32")
    weight = np.zeros((ph, pw), dtype="float32")
    stride = max(1, tile_size - overlap)
    device = torch.device(device)
    use_amp = amp_dtype is not None
    for r, col in tile_windows(ph, pw, tile_size, stride):
        x = torch.from_numpy(img[None, :, r : r + tile_size, col : col + tile_size]).to(device)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
            logits = model(x)
        p = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        prob[r : r + tile_size, col : col + tile_size] += p
        weight[r : r + tile_size, col : col + tile_size] += 1.0
    return (prob / np.maximum(weight, 1e-6))[:h, :w]
