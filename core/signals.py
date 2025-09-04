"""Signal processing helpers for Peakfit 3.x.

Currently provides stubs for baseline correction and S/N estimation.
"""
from __future__ import annotations

import numpy as np

__all__ = ["als_baseline", "snr_estimate", "polynomial_baseline"]


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


def polynomial_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 2,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    normalize_x: bool = True,
) -> np.ndarray:
    """Weighted least-squares polynomial baseline on ``(x, y)``.

    Fits on masked points if ``mask`` is provided and returns the baseline
    evaluated on the full ``x`` domain.  ``weights`` apply to the fit region
    only.  When ``normalize_x`` is ``True`` the fit is performed on scaled
    coordinates in ``[-1, 1]`` for improved numerical stability.
    """

    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if degree < 0:
        raise ValueError("degree must be >= 0")

    if mask is not None:
        m = np.asarray(mask, bool)
        x_fit, y_fit = x[m], y[m]
        w_fit = None if weights is None else np.asarray(weights, float)[m]
    else:
        x_fit, y_fit = x, y
        w_fit = None if weights is None else np.asarray(weights, float)

    if normalize_x:
        x_min = float(x_fit.min())
        x_max = float(x_fit.max())
        span = max(x_max - x_min, 1e-12)

        def scale(xx):
            return 2.0 * (xx - x_min) / span - 1.0

        xf = scale(x_fit)
        x_all = scale(x)
    else:
        xf = x_fit
        x_all = x

    V = np.vander(xf, N=degree + 1, increasing=True)
    if w_fit is not None:
        w = np.sqrt(np.clip(w_fit, 0.0, np.inf))[:, None]
        A = V * w
        b = y_fit[:, None] * w
    else:
        A = V
        b = y_fit[:, None]

    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    V_all = np.vander(x_all, N=degree + 1, increasing=True)
    return (V_all @ beta).ravel()
