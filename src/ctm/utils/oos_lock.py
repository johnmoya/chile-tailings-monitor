"""Single out-of-sample run guard for the test deposits.

The lock is created atomically (O_EXCL) BEFORE any test data is read, so an
evaluation that crashes half-way still counts as "the" run. Deleting the lock
is a deliberate, visible act (it lives in reports/ and is committed to git).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

CONFIRM_FLAG = "--i-understand-this-is-the-single-oos-run"


class OOSLockError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd: str | Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def read_lock(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"status": "unreadable"}


def acquire(path: str | Path, info: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "started", "started_at": datetime.now(UTC).isoformat(), **info}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as exc:
        raise OOSLockError(f"test evaluation already ran: {path}") from exc
    with os.fdopen(fd, "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def update(path: str | Path, **fields) -> dict:
    path = Path(path)
    record = read_lock(path) or {}
    record.update(fields)
    os.chmod(path, 0o644)
    path.write_text(json.dumps(record, indent=2))
    os.chmod(path, 0o444)
    return record
