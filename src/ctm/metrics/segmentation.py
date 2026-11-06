"""Binary segmentation metrics over labelled pixels (ignore_index excluded).

Counts are accumulated (TP/FP/FN/TN) so that dataset-level IoU/F1 is the
micro-average over pixels, not the mean of per-tile scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __iadd__(self, other: Confusion) -> Confusion:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    def as_dict(self) -> dict:
        return {"iou": self.iou, "f1": self.f1, "precision": self.precision,
                "recall": self.recall, "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def confusion(pred, target, valid=None) -> Confusion:
    """``pred``/``target`` boolean-like arrays (numpy or torch); ``valid`` selects pixels."""
    try:
        import torch

        if isinstance(pred, torch.Tensor):
            pred, target = pred.bool(), target.bool()
            v = torch.ones_like(pred) if valid is None else valid.bool()
            return Confusion(
                tp=int((pred & target & v).sum()), fp=int((pred & ~target & v).sum()),
                fn=int((~pred & target & v).sum()), tn=int((~pred & ~target & v).sum()),
            )
    except ImportError:  # pragma: no cover
        pass
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    v = np.ones_like(pred) if valid is None else np.asarray(valid, bool)
    return Confusion(
        tp=int((pred & target & v).sum()), fp=int((pred & ~target & v).sum()),
        fn=int((~pred & target & v).sum()), tn=int((~pred & ~target & v).sum()),
    )


def deposit_summary(per_deposit: dict[str, Confusion]) -> dict:
    """PRIMARY metric: IoU per deposit (pixels pooled within each deposit), then the
    unweighted mean across deposits. SECONDARY: IoU over all pixels pooled.

    Deposits whose IoU is undefined (no water in reference nor prediction) are
    excluded from the macro mean and counted in ``n_deposits_undefined``.
    Both numbers are always reported together.
    """
    ious = {d: cm.iou for d, cm in per_deposit.items()}
    f1s = {d: cm.f1 for d, cm in per_deposit.items()}
    defined = [v for v in ious.values() if v == v]
    defined_f1 = [v for v in f1s.values() if v == v]
    pooled = Confusion()
    for cm in per_deposit.values():
        pooled += cm
    return {
        "iou_macro_deposit": float(np.mean(defined)) if defined else float("nan"),
        "f1_macro_deposit": float(np.mean(defined_f1)) if defined_f1 else float("nan"),
        "iou_micro_pixel": pooled.iou,
        "f1_micro_pixel": pooled.f1,
        "precision_micro_pixel": pooled.precision,
        "recall_micro_pixel": pooled.recall,
        "n_deposits": len(per_deposit),
        "n_deposits_undefined": len(per_deposit) - len(defined),
        "iou_per_deposit": ious,
    }


def iou(pred, target, valid=None) -> float:
    return confusion(pred, target, valid).iou


def f1(pred, target, valid=None) -> float:
    return confusion(pred, target, valid).f1
