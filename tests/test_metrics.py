import math

import numpy as np
import pytest
import torch

from ctm.metrics.segmentation import Confusion, confusion, deposit_summary


def test_confusion_numpy_and_torch_agree():
    rng = np.random.default_rng(0)
    p, t, v = (rng.random((32, 32)) > 0.5 for _ in range(3))
    a = confusion(p, t, v)
    b = confusion(torch.from_numpy(p), torch.from_numpy(t), torch.from_numpy(v))
    assert a == b
    assert a.tp + a.fp + a.fn + a.tn == v.sum()


def test_iou_f1_values():
    cm = Confusion(tp=6, fp=2, fn=2, tn=90)
    assert cm.iou == pytest.approx(0.6)
    assert cm.f1 == pytest.approx(0.75)


def test_empty_is_nan():
    assert math.isnan(Confusion().iou)


def test_macro_by_deposit_differs_from_micro_by_pixel():
    # Big deposit, perfect; small deposit, poor. Micro is dominated by the big one.
    per_dep = {
        "big": Confusion(tp=9000, fp=0, fn=0, tn=1000),
        "small": Confusion(tp=10, fp=20, fn=20, tn=50),
    }
    s = deposit_summary(per_dep)
    assert s["iou_macro_deposit"] == pytest.approx((1.0 + 0.2) / 2)
    assert s["iou_micro_pixel"] == pytest.approx(9010 / 9050)
    assert s["n_deposits"] == 2 and s["n_deposits_undefined"] == 0
    assert set(s["iou_per_deposit"]) == {"big", "small"}


def test_macro_excludes_undefined_deposits():
    s = deposit_summary({"a": Confusion(tp=5, fp=5, fn=0), "dry": Confusion(tn=100)})
    assert s["iou_macro_deposit"] == pytest.approx(0.5)
    assert s["n_deposits_undefined"] == 1
