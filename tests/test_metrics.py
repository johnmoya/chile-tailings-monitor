import math

import numpy as np
import pytest
import torch

from ctm.metrics.segmentation import Confusion, confusion


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
