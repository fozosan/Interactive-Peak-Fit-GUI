"""Signal processing helpers for Peakfit 3.x.

Currently provides stubs for baseline correction and S/N estimation.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["als_baseline", "snr_estimate", "polynomial_baseline"]


def _second_difference_gram_bands(L: int) -> np.ndarray:
    """``D.T @ D`` for the (L-2) x L second-difference matrix ``D``, as bands.

    The product is symmetric and pentadiagonal. The result is laid out for
    ``solve_banded((2, 2), ...)``: element (i, j) sits at ``[2 + i - j, j]``.
    """

    bands = np.zeros((5, L))
    stencil = (1.0, -2.0, 1.0)
    rows = L - 2
    if rows > 0:
        # Row k of D holds the stencil in columns k..k+2, so it adds
        # stencil[a] * stencil[b] to element (k + a, k + b).
        for a in range(3):
            for b in range(3):
                bands[2 + a - b, b:b + rows] += stencil[a] * stencil[b]
    return bands


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001,
                 niter: int = 10, tol: float = 0.0) -> np.ndarray:
    """Estimate an asymmetric least squares (ALS) baseline for ``y``.

    Parameters mirror the blueprint description. The implementation follows
    Eilers & Boelens (2005), iteratively updating weights for points above the
    baseline. ``lam`` controls smoothness while ``p`` controls asymmetry.

    Each iteration solves ``(W + lam * D.T @ D) z = w * y``. That matrix is
    pentadiagonal, so it is solved in banded form: a 1650-point baseline takes
    about a millisecond, where the dense solve took seconds.
    """

    y = np.asarray(y, dtype=float)
    L = y.size
    smooth = lam * _second_difference_gram_bands(L)  # loop-invariant
    w = np.ones(L)
    z = np.zeros_like(y)
    for _ in range(int(niter)):
        A = smooth.copy()
        A[2] += w  # W is diagonal
        Z = solve_banded((2, 2), A, w * y, overwrite_ab=True)
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
    return_used_degree: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Weighted least-squares polynomial baseline on ``(x, y)``.

    Fits on masked points if ``mask`` is provided and returns the baseline
    evaluated on the full ``x`` domain.  ``weights`` apply to the fit region
    only.  When ``normalize_x`` is ``True`` the fit is performed on scaled
    coordinates in ``[-1, 1]`` for improved numerical stability.
    """

    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if mask is not None:
        m = np.asarray(mask, bool)
        x_fit, y_fit = x[m], y[m]
        w_fit = None if weights is None else np.asarray(weights, float)[m]
    else:
        x_fit, y_fit = x, y
        w_fit = None if weights is None else np.asarray(weights, float)

    max_deg = max(0, int(x_fit.size) - 1)
    used_deg = min(max(0, int(degree)), max_deg)

    if normalize_x:
        x_min = float(x_fit.min()) if x_fit.size else 0.0
        x_max = float(x_fit.max()) if x_fit.size else 1.0
        span = max(x_max - x_min, 1e-12)

        def scale(xx: np.ndarray) -> np.ndarray:
            return 2.0 * (xx - x_min) / span - 1.0

        xf = scale(x_fit)
        x_all = scale(x)
    else:
        xf = x_fit
        x_all = x

    V = np.vander(xf, N=used_deg + 1, increasing=True)
    if w_fit is not None:
        w = np.sqrt(np.clip(w_fit, 0.0, np.inf))[:, None]
        A = V * w
        b = y_fit[:, None] * w
    else:
        A = V
        b = y_fit[:, None]

    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    V_all = np.vander(x_all, N=used_deg + 1, increasing=True)
    baseline = (V_all @ beta).ravel()
    return (baseline, used_deg) if return_used_degree else baseline
