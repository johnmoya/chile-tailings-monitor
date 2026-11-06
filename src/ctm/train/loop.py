"""Training loop: bf16 autocast, fixed seeds, masked BCE+Dice, checkpoints, JSONL metric log."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ctm.metrics.segmentation import Confusion, confusion
from ctm.models.unet import build_model
from ctm.utils.seed import seed_everything, worker_init_fn

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    best_val_iou: float = float("nan")
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    peak_vram_mib: float | None = None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, ignore_index: int,
                    bce_weight: float = 0.5, smooth: float = 1.0) -> torch.Tensor:
    """logits (N,1,H,W), target (N,H,W) int with ignore_index; loss over labelled pixels."""
    logits = logits[:, 0].float()
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], t[valid])
    p = torch.sigmoid(logits) * valid
    tv = t * valid
    inter = (p * tv).sum()
    dice = 1.0 - (2 * inter + smooth) / (p.sum() + tv.sum() + smooth)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def _loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int,
            device: torch.device) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=g, worker_init_fn=worker_init_fn,
                      pin_memory=device.type == "cuda", drop_last=False,
                      persistent_workers=workers > 0)


def _run_epoch(model, loader, device, amp, amp_dtype, ignore_index, bce_weight, threshold,
               optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches, cm = 0.0, 0, Confusion()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), \
                torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = model(x)
            loss = masked_bce_dice(logits, y, ignore_index, bce_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
        with torch.no_grad():
            pred = torch.sigmoid(logits[:, 0].float()) > threshold
            cm += confusion(pred, y == 1, y != ignore_index)
    return {"loss": total_loss / max(n_batches, 1), **cm.as_dict()}


def train(cfg: dict, train_ds: Dataset, val_ds: Dataset | None, run_dir: str | Path) -> TrainResult:
    tcfg, ignore_index = cfg["train"], cfg["labels"]["ignore_index"]
    seed_everything(cfg["seed"], deterministic=tcfg.get("deterministic", True))
    device = resolve_device(tcfg["device"])
    amp = bool(tcfg["amp"])
    amp_dtype = AMP_DTYPES[tcfg["amp_dtype"]]
    if amp and device.type == "cuda" and amp_dtype == torch.bfloat16 \
            and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 AMP requested but this GPU does not support bf16")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log_path = run_dir / "metrics.jsonl"

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    train_loader = _loader(train_ds, tcfg["batch_size"], True, tcfg["num_workers"],
                           cfg["seed"], device)
    val_loader = (_loader(val_ds, tcfg["batch_size"], False, tcfg["num_workers"],
                          cfg["seed"], device) if val_ds is not None else None)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = TrainResult()
    best = -1.0
    common = dict(device=device, amp=amp, amp_dtype=amp_dtype, ignore_index=ignore_index,
                  bce_weight=tcfg["bce_weight"], threshold=tcfg["threshold"])
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer=optimizer, **common)
        record = {"epoch": epoch, "train": tr, "seconds": None}
        if val_loader is not None:
            record["val"] = _run_epoch(model, val_loader, **common)
        record["seconds"] = round(time.perf_counter() - t0, 3)
        if device.type == "cuda":
            record["peak_vram_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        result.history.append(record)

        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "metrics": record,
                 "model_cfg": cfg["model"], "seed": cfg["seed"]}
        result.last_checkpoint = run_dir / "last.pt"
        torch.save(state, result.last_checkpoint)
        score = record.get("val", record["train"])["iou"]
        if score == score and score > best:  # NaN-safe
            best = score
            result.best_val_iou = score
            result.best_checkpoint = run_dir / "best.pt"
            torch.save(state, result.best_checkpoint)

    if device.type == "cuda":
        result.peak_vram_mib = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(state["model_cfg"], encoder_weights=None)  # weights come from the checkpoint
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    return model.to(device), state
