"""Signal processing helpers for Peakfit 3.x.

Currently provides stubs for baseline correction and S/N estimation.
"""
from __future__ import annotations

import numpy as np


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001,
                 niter: int = 10, tol: float = 0.0) -> np.ndarray:
    """Estimate an asymmetric least squares (ALS) baseline for ``y``.

    Parameters mirror the blueprint description. The implementation follows
    Eilers & Boelens (2005), iteratively updating weights for points above the
    baseline. ``lam`` controls smoothness while ``p`` controls asymmetry.
    """

    y = np.asarray(y, dtype=float)
    L = y.size
    # second-order difference matrix; diff along rows to keep square result
    D = np.diff(np.eye(L), 2, axis=0)
    w = np.ones(L)
    z = np.zeros_like(y)
    for _ in range(int(niter)):
        W = np.diag(w)
        Z = np.linalg.solve(W + lam * D.T @ D, w * y)
        if tol and np.sqrt(np.mean((Z - z) ** 2)) < tol:
            z = Z
            break
        z = Z
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def snr_estimate(y: np.ndarray) -> float:
    """Return a robust signal-to-noise estimate for ``y``.

    Noise is estimated from the median absolute deviation of successive
    differences, while signal is taken as the 90th percentile minus the median
    of ``y``. The returned value is ``signal / noise``.
    """

    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return float("nan")
    noise = np.median(np.abs(np.diff(y))) / 0.6745
    signal = np.percentile(y, 90) - np.median(y)
    if noise == 0:
        return float("inf")
    return float(signal / noise)
