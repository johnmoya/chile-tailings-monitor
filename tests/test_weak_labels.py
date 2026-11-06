import numpy as np
import pytest

from ctm.labels.weak import (
    CorrectionHook,
    ManualCorrectionStore,
    WeakLabel,
    make_weak_label,
)
from tests.test_baseline import synthetic_pond


def _scene(cfg):
    green, swir, water = synthetic_pond()
    bands = cfg["stac"]["bands"]
    image = np.full((len(bands), *green.shape), 0.2, dtype="float32")
    image[bands.index("B03")], image[bands.index("B11")] = green, swir
    poly = np.zeros_like(water)
    poly[10:110, 10:150] = True
    valid = np.ones_like(water)
    valid[10:20, :] = False
    return image, bands, poly, valid, water


def test_weak_label_encoding(cfg):
    image, bands, poly, valid, water = _scene(cfg)
    ign = cfg["labels"]["ignore_index"]
    lab = make_weak_label(image, bands, poly, "dep-a", "scene-1", cfg["baseline"], ign, valid)
    assert lab.mask.dtype == np.uint8
    assert set(np.unique(lab.mask)) <= {0, 1, ign}
    assert (lab.mask[~poly] == ign).all()
    assert (lab.mask[~valid] == ign).all()
    scored = lab.mask != ign
    assert ((lab.mask == 1) == water)[scored].mean() > 0.99
    assert lab.source == "mndwi_otsu" and lab.corrections == []


def test_correction_hook_interface(cfg):
    class FlipCorner(CorrectionHook):
        name = "flip-corner"

        def apply(self, label: WeakLabel) -> WeakLabel:
            mask = label.mask.copy()
            mask[50:55, 50:55] = 0
            return WeakLabel(mask, label.deposit_id, label.scene_id, label.source,
                             label.threshold, label.fallback_used,
                             label.corrections + [self.name])

    image, bands, poly, valid, _ = _scene(cfg)
    lab = make_weak_label(image, bands, poly, "dep-a", "s", cfg["baseline"], valid=valid,
                          hooks=[FlipCorner()])
    assert lab.corrections == ["flip-corner"]
    assert (lab.mask[50:55, 50:55] == 0).all()


def test_manual_store_is_interface_only(cfg):
    image, bands, poly, valid, _ = _scene(cfg)
    with pytest.raises(NotImplementedError):
        make_weak_label(image, bands, poly, "d", "s", cfg["baseline"],
                        hooks=[ManualCorrectionStore("unused")])
