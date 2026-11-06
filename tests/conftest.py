from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ctm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # scripts are importable as modules in tests


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def tmp_cfg(tmp_path):
    """Default config with every data/report path redirected into tmp_path."""
    return load_config(overrides={
        "paths.scenes_dir": str(tmp_path / "scenes"),
        "paths.tiles_dir": str(tmp_path / "tiles"),
        "paths.runs_dir": str(tmp_path / "runs"),
        "paths.reports_dir": str(tmp_path / "reports"),
        "paths.test_lock": str(tmp_path / "reports" / "locks" / "test.lock"),
        "paths.deposits": str(tmp_path / "deposits.geojson"),
    })
