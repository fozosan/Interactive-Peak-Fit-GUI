"""Utility helpers for solvers."""
from __future__ import annotations

import numpy as np


def mad_sigma(residual: np.ndarray) -> float:
    """Estimate noise scale using the MAD of ``residual``."""
    r = np.asarray(residual, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return 1.4826 * mad


def robust_cost(r: np.ndarray, loss: str, f_scale: float) -> float:
    """Return robust cost matching :func:`scipy.optimize.least_squares` losses."""
    r = np.asarray(r, dtype=float)
    fs = float(max(f_scale, 1e-12))
    z = r / fs
    if loss == "linear":
        rho = z**2
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z**2) - 1.0)
    elif loss == "huber":
        mask = np.abs(z) <= 1.0
        rho = np.where(mask, z**2, 2.0 * np.abs(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z**2)
    else:
        raise ValueError(f"unknown loss '{loss}'")
    return 0.5 * fs**2 * float(np.sum(rho))
