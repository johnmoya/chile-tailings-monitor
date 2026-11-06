"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch; optionally request deterministic kernels.

    ``deterministic=True`` uses ``warn_only`` so ops without a deterministic
    implementation warn instead of crashing.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from torch's per-worker base seed."""
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
