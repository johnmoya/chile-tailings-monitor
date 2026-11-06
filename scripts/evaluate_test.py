"""THE single out-of-sample evaluation on the TEST deposits.

Refuses to run without --i-understand-this-is-the-single-oos-run. All
preconditions are checked BEFORE the lock is taken, so a missing checkpoint
or missing reference mask does not burn the run. Once the lock exists this
script refuses forever (exit 3).

Exit codes: 0 ok, 2 missing confirmation flag, 3 already ran (lock present),
4 preconditions not met (no lock written), 5 evaluation failed (lock kept).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from ctm.data.deposits import load_deposits
from ctm.data.scene_io import load_scene_npz
from ctm.utils import oos_lock
from ctm.utils.config import load_config, resolve_path


def _preconditions(cfg: dict, checkpoint: Path) -> tuple[list[str], dict[str, list[Path]]]:
    problems: list[str] = []
    files: dict[str, list[Path]] = {}
    if not checkpoint.is_file():
        problems.append(f"checkpoint not found: {checkpoint}")
    try:
        reg = load_deposits(resolve_path(cfg, "deposits"), allow_placeholders=False)
    except Exception as exc:  # schema errors, placeholders, missing file
        return problems + [f"deposits: {exc}"], files
    test_ids = reg.ids_for("test", allow_test=True)
    if not test_ids:
        problems.append("no deposits with split == 'test'")
    scenes_dir = resolve_path(cfg, "scenes_dir")
    for did in test_ids:
        found = sorted((scenes_dir / did).glob("*.npz"))
        if not found:
            problems.append(f"no scenes for test deposit {did} in {scenes_dir / did}")
        files[did] = found
    return problems, files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="results JSON (default <reports_dir>/test_results.json)")
    p.add_argument(oos_lock.CONFIRM_FLAG, dest="confirmed", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    lock_path = resolve_path(cfg, "test_lock")

    if not args.confirmed:
        print(f"Refusing to run: the test deposits are evaluated exactly once.\n"
              f"Re-run with {oos_lock.CONFIRM_FLAG} when the model is FINAL.", file=sys.stderr)
        return 2
    existing = oos_lock.read_lock(lock_path)
    if existing is not None:
        print(f"Refusing to run: test evaluation already happened ({lock_path}).\n"
              f"{json.dumps(existing, indent=2)}", file=sys.stderr)
        return 3
    problems, files = _preconditions(cfg, args.checkpoint)
    if problems:
        print("Preconditions not met (no lock written):\n  - " + "\n  - ".join(problems),
              file=sys.stderr)
        return 4

    out = args.out or resolve_path(cfg, "reports_dir") / "test_results.json"
    try:
        oos_lock.acquire(lock_path, {
            "git_commit": oos_lock.git_commit(cfg["_root"]),
            "config": cfg["_source"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": oos_lock.sha256_file(args.checkpoint),
            "test_deposits": sorted(files),
            "n_scenes": sum(len(v) for v in files.values()),
        })
    except oos_lock.OOSLockError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        return 3

    try:
        import torch

        from ctm.metrics.evaluation import evaluate_scenes
        from ctm.train.loop import load_checkpoint, resolve_device

        device = resolve_device(cfg["train"]["device"])
        model, _ = load_checkpoint(args.checkpoint, device)
        scenes = (load_scene_npz(f) for fs in files.values() for f in fs)
        with torch.inference_mode():
            results = evaluate_scenes(model, scenes, cfg, device)
        results["meta"] = {"finished_at": datetime.now(UTC).isoformat(),
                           "lock": str(lock_path)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        oos_lock.update(lock_path, status="completed", results=str(out),
                        results_sha256=oos_lock.sha256_file(out),
                        finished_at=results["meta"]["finished_at"])
    except Exception as exc:
        oos_lock.update(lock_path, status="failed", error=repr(exc),
                        traceback=traceback.format_exc())
        print(f"Evaluation failed; the lock is kept ({lock_path}).", file=sys.stderr)
        raise SystemExit(5) from exc

    overall = results["overall"]
    for name in ("baseline", "unet"):
        o = overall[name]
        print(f"{name:9s} IoU macro/deposit (PRIMARY)={o['iou_macro_deposit']:.4f} "
              f"IoU micro/pixel={o['iou_micro_pixel']:.4f} "
              f"F1 macro={o['f1_macro_deposit']:.4f} area MAE={o['area_mae_m2']:.0f} m2 "
              f"(deposits={o['n_deposits']}, undefined={o['n_deposits_undefined']})")
    print(f"results -> {out}\nlock    -> {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
