"""Scene-level comparison of U-Net vs MNDWI+Otsu against manual reference masks.

Both methods are scored on exactly the same pixels: inside the polygon, valid
(cloud-free) and labelled in the reference. The baseline runs on the whole
polygon (Otsu needs the full histogram), the model by sliding window.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch

from ctm.baseline.mndwi_otsu import segment_scene
from ctm.data.scene_io import StoredScene
from ctm.metrics.segmentation import Confusion, confusion, deposit_summary
from ctm.models.inference import predict_scene


def evaluate_scenes(
    model: torch.nn.Module,
    scenes: Iterable[StoredScene],
    cfg: dict,
    device: torch.device | str = "cpu",
) -> dict:
    ignore = cfg["labels"]["ignore_index"]
    thr = cfg["train"]["threshold"]
    amp_dtype = torch.bfloat16 if cfg["train"]["amp"] and torch.device(device).type == "cuda" \
        else None
    methods = ("unet", "baseline")
    per_scene: list[dict] = []
    # per_deposit[method][deposit_id] -> pooled confusion of that deposit's scenes
    per_deposit: dict[str, dict[str, Confusion]] = {m: defaultdict(Confusion) for m in methods}

    for sc in scenes:
        if sc.reference is None:
            raise ValueError(f"scene {sc.meta.get('scene_id')} has no reference mask")
        valid = sc.valid
        scored = sc.polygon_mask & valid & (sc.reference != ignore)
        ref = sc.reference == 1
        prob = predict_scene(model, sc.image, cfg["inference"]["tile_size"],
                             cfg["inference"]["overlap"], device, amp_dtype)
        preds = {
            "unet": prob > thr,
            "baseline": segment_scene(sc.image, sc.bands, sc.polygon_mask, cfg["baseline"],
                                      valid=valid).water,
        }
        px_area = float(sc.meta.get("pixel_area_m2", 100.0))
        rec = {"deposit_id": sc.meta["deposit_id"], "scene_id": sc.meta.get("scene_id"),
               "scored_pixels": int(scored.sum()),
               "ref_area_m2": float((ref & scored).sum() * px_area)}
        for name, pred in preds.items():
            cm = confusion(pred, ref, scored)
            per_deposit[name][sc.meta["deposit_id"]] += cm
            area = float((pred & scored).sum() * px_area)
            rec[name] = {"iou": cm.iou, "f1": cm.f1, "area_m2": area,
                         "area_abs_err_m2": abs(area - rec["ref_area_m2"])}
        per_scene.append(rec)

    def _area_mae(name: str) -> float:
        errs = [r[name]["area_abs_err_m2"] for r in per_scene]
        return float(np.mean(errs)) if errs else float("nan")

    return {
        # PRIMARY: iou_macro_deposit. SECONDARY: iou_micro_pixel (+ the rest).
        "overall": {m: dict(deposit_summary(per_deposit[m]), area_mae_m2=_area_mae(m))
                    for m in methods},
        "per_deposit": {m: {d: cm.as_dict() for d, cm in per_deposit[m].items()}
                        for m in methods},
        "per_scene": per_scene,
    }
