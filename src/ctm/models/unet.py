"""U-Net (segmentation_models_pytorch) with a ResNet34 encoder and 6 input bands."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class Normalize(nn.Module):
    """Per-band standardisation; identity when mean/std are not configured.

    Stats must come from TRAIN deposits only (computed in M1).
    """

    def __init__(self, mean: list[float] | None, std: list[float] | None, channels: int):
        super().__init__()
        m = torch.zeros(channels) if mean is None else torch.tensor(mean, dtype=torch.float32)
        s = torch.ones(channels) if std is None else torch.tensor(std, dtype=torch.float32)
        if m.numel() != channels or s.numel() != channels:
            raise ValueError("norm_mean/norm_std length must equal in_channels")
        self.register_buffer("mean", m.view(1, -1, 1, 1))
        self.register_buffer("std", s.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class WaterSegmenter(nn.Module):
    def __init__(self, net: nn.Module, norm: Normalize):
        super().__init__()
        self.norm = norm
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))   # logits (N, 1, H, W)


def build_model(model_cfg: dict) -> WaterSegmenter:
    if model_cfg.get("arch", "unet") != "unet":
        raise ValueError(f"unsupported arch {model_cfg['arch']!r}")
    net = smp.Unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )
    norm = Normalize(model_cfg.get("norm_mean"), model_cfg.get("norm_std"),
                     model_cfg["in_channels"])
    return WaterSegmenter(net, norm)
