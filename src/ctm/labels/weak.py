"""Weak labels from the MNDWI + Otsu baseline, plus a hook for manual corrections.

Label encoding (uint8): 0 = not water, 1 = water, ``ignore_index`` = not
evaluated (outside polygon, cloud/shadow/snow, no data). The ignore pixels
are excluded from the training loss and from metrics.

Manual corrections are NOT implemented in M0: :class:`CorrectionHook` only
fixes the interface so that M1+ can plug in edits (e.g. polygons drawn in
QGIS) without touching the training pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ctm.baseline.mndwi_otsu import BaselineResult, segment_scene


@dataclass
class WeakLabel:
    mask: np.ndarray             # uint8 (H, W)
    deposit_id: str
    scene_id: str
    source: str = "mndwi_otsu"
    threshold: float | None = None
    fallback_used: bool = False
    corrections: list[str] = field(default_factory=list)   # provenance of applied hooks


class CorrectionHook(ABC):
    """Interface for manual corrections applied on top of a weak label.

    Implementations must:
      * return a NEW ``WeakLabel`` (never mutate the input in place);
      * only change pixels they have an explicit opinion about;
      * append an identifier to ``corrections`` for provenance.
    """

    name: str = "correction"

    @abstractmethod
    def apply(self, label: WeakLabel) -> WeakLabel: ...


class NoCorrection(CorrectionHook):
    name = "none"

    def apply(self, label: WeakLabel) -> WeakLabel:
        return label


class ManualCorrectionStore(CorrectionHook):
    """Placeholder for corrections loaded from disk (format to be decided in M1+)."""

    name = "manual"

    def __init__(self, root):
        self.root = root

    def apply(self, label: WeakLabel) -> WeakLabel:
        raise NotImplementedError("manual corrections are planned after M0")


def from_baseline(result: BaselineResult, deposit_id: str, scene_id: str,
                  ignore_index: int = 255) -> WeakLabel:
    mask = np.full(result.water.shape, ignore_index, dtype="uint8")
    mask[result.evaluated] = result.water[result.evaluated].astype("uint8")
    return WeakLabel(mask, deposit_id, scene_id, threshold=result.threshold,
                     fallback_used=result.fallback_used)


def make_weak_label(
    image: np.ndarray,
    bands: list[str],
    polygon_mask: np.ndarray,
    deposit_id: str,
    scene_id: str,
    baseline_cfg: dict,
    ignore_index: int = 255,
    valid: np.ndarray | None = None,
    hooks: list[CorrectionHook] | None = None,
) -> WeakLabel:
    result = segment_scene(image, bands, polygon_mask, baseline_cfg, valid=valid)
    label = from_baseline(result, deposit_id, scene_id, ignore_index)
    for hook in hooks or [NoCorrection()]:
        label = hook.apply(label)
    return label
