"""Performance configuration for Peakfit 3.x."""
from __future__ import annotations

from typing import Optional


def set_numba(flag: bool) -> None:
    """Enable or disable Numba acceleration."""
    raise NotImplementedError("Numba toggle not yet implemented")


def set_gpu(flag: bool) -> None:
    """Enable or disable CuPy GPU acceleration."""
    raise NotImplementedError("GPU toggle not yet implemented")


def set_seed(seed: Optional[int]) -> None:
    """Seed all random number generators for determinism."""
    raise NotImplementedError("Seed control not yet implemented")


def set_max_workers(n: int) -> None:
    """Set the maximum number of parallel workers."""
    raise NotImplementedError("Worker configuration not yet implemented")
