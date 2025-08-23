"""LMFIT backend providing parameter constraints and alternative algorithms."""
from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np


class SolveResult(TypedDict):
    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict


def solve(x: np.ndarray, y: np.ndarray, peaks: list, mode: str,
          baseline: np.ndarray | None, options: dict) -> SolveResult:
    """Fit peaks using the optional LMFIT dependency."""
    raise NotImplementedError("LMFIT backend not yet implemented")