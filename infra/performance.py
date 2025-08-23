"""Performance configuration for Peakfit 3.x."""
from __future__ import annotations
from typing import Optional
import numpy as np
import random
from core import models

_numba_enabled = False
_gpu_enabled = False
_max_workers = 0


def set_numba(flag: bool) -> None:
    """Enable or disable Numba acceleration if available."""

    global _numba_enabled
    try:  # pragma: no cover - optional dependency
        import numba  # noqa: F401

        _numba_enabled = bool(flag)
    except Exception:
        _numba_enabled = False


def set_gpu(flag: bool) -> None:
    """Enable or disable CuPy GPU acceleration."""

    global _gpu_enabled
    if flag and models.cp is not None:
        models.xp = models.cp
        _gpu_enabled = True
    else:
        models.xp = np
        _gpu_enabled = False


def set_seed(seed: Optional[int]) -> None:
    """Seed all random number generators for determinism."""

    np.random.seed(seed if seed is not None else None)
    random.seed(seed)
    if models.cp is not None:
        try:  # pragma: no cover - CuPy may be absent
            models.cp.random.seed(seed)
        except Exception:
            pass


def set_max_workers(n: int) -> None:
    """Set the maximum number of parallel workers."""

    global _max_workers
    _max_workers = max(0, int(n))