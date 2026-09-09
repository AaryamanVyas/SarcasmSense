"""Deterministic seeding helpers.

Reproducibility is a hard requirement of this project (README section
"Reproducibility"), so every entry-point calls :func:`set_global_seed` exactly
once, as early as possible, before any model or dataloader is constructed.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    """Record of what was seeded, persisted into experiment metadata."""

    seed: int
    deterministic_torch: bool
    cudnn_deterministic: bool


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) and hash randomisation.

    Args:
        seed: The integer seed to apply everywhere.
        deterministic: When True, ask cuDNN/PyTorch for deterministic kernels.
            This costs some throughput but makes runs bit-comparable, which is
            what we want for an academic evaluation.

    Returns:
        A :class:`SeedState` describing what was actually applied. PyTorch is
        imported lazily so that pure-data utilities do not need to pay the
        import cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    cudnn_deterministic = False
    torch_deterministic = False
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # ``warn_only`` keeps us from hard-crashing on ops that have no
            # deterministic implementation (e.g. some scatter ops on CUDA).
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch_deterministic = True
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                cudnn_deterministic = True
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        logger.warning("PyTorch not installed; only Python/NumPy seeds were set.")

    logger.debug("Global seed set to %d", seed)
    return SeedState(
        seed=seed,
        deterministic_torch=torch_deterministic,
        cudnn_deterministic=cudnn_deterministic,
    )


def seed_worker(worker_id: int) -> None:  # pragma: no cover - exercised by DataLoader
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`.

    Each worker derives its seed from the torch initial seed so that shuffling
    stays reproducible across multi-process data loading.
    """
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
