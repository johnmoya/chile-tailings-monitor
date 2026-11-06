"""Check annotation masks: grid (vs rgb/falsecolor), dtype, values {0,1,255}, coverage.

Status per scene: ok | incomplete (coverage < annotation.min_coverage) |
pending (untouched template) | error (grid, dtype, values, missing or forbidden files).

Exit 1 if any scene is in error; with ``--require-complete`` also if any is
pending or incomplete. Usage: ``uv run python scripts/validate_annotations.py``
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ctm.labels.annotation import validate_scene_dir
from ctm.utils.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--root", default=None, help="annotation dir (default from config)")
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(args.root) if args.root else resolve_path(cfg, "annotation_dir")
    dirs = sorted(d for d in root.glob("*/*") if d.is_dir())
    if not dirs:
        print(f"no scene folders under {root}", file=sys.stderr)
        return 1
    min_cov = cfg["annotation"]["min_coverage"]
    reports = [validate_scene_dir(d, min_cov) for d in dirs]
    print(f"{'scene':40s} {'status':10s} {'coverage':>8s} {'water':>6s}  problems")
    for r in reports:
        rel = r.path.relative_to(root)
        cov = "" if r.coverage != r.coverage else f"{100 * r.coverage:.1f}%"
        wat = "" if r.water_fraction != r.water_fraction else f"{100 * r.water_fraction:.1f}%"
        print(f"{str(rel):40s} {r.status:10s} {cov:>8s} {wat:>6s}  {'; '.join(r.problems)}")
    counts = Counter(r.status for r in reports)
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if counts.get("error"):
        return 1
    if args.require_complete and (counts.get("pending") or counts.get("incomplete")):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
