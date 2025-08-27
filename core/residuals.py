"""Residual and Jacobian construction for Peakfit 3.x."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
try:  # optional CuPy support
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy may be absent
    cp = None  # type: ignore

from .jacobians import pv_and_grads
from .peaks import Peak
from infra import performance as perf


def build_residual(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    mode: str,
    baseline: np.ndarray | None,
    loss: str,
    weights: np.ndarray | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a residual function ``r(theta)`` for the given data.

    ``theta`` is expected to contain ``4 * len(peaks)`` parameters ordered as
    ``(center, height, fwhm, eta)`` for each peak. ``loss`` is currently
    accepted for API compatibility but only influences callers that apply a
    robust loss externally. ``weights`` can provide per-point weighting.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None
    w = np.asarray(weights, dtype=float) if weights is not None else None

    def residual(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.size != 4 * len(peaks):
            raise ValueError("theta has wrong size")
        pk = []
        for i in range(len(peaks)):
            c, h, fw, eta = theta[4 * i : 4 * (i + 1)]
            pk.append((h, c, fw, eta))
        model = perf.eval_total(x, pk)
        base = baseline if baseline is not None else 0.0
        if mode == "add":
            r = model + base - y
        elif mode == "subtract":
            r = model - (y - base)
        else:  # pragma: no cover - unknown mode
            raise ValueError("unknown mode")
        if w is not None:
            r = r * w
        return r

    return residual


def build_residual_jac(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    mode: str,
    baseline: np.ndarray | None,
    weights: np.ndarray | None,
    wmin_eval: float = 0.0,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return residual and Jacobian builder for solvers.

    The parameter vector ``theta`` follows the order ``[h1,(c1),(w1), h2,(c2),(w2), ...]``
    where centres and widths are omitted when locked.  ``wmin_eval`` provides a
    minimum width used only for evaluation to guard against singular intermediate
    values.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None
    w = np.asarray(weights, dtype=float) if weights is not None else None

    n = len(peaks)
    h0 = np.array([p.height for p in peaks], dtype=float)
    c0 = np.array([p.center for p in peaks], dtype=float)
    f0 = np.array([p.fwhm for p in peaks], dtype=float)
    e = np.array([p.eta for p in peaks], dtype=float)
    lock_c = np.array([p.lock_center for p in peaks], dtype=bool)
    lock_w = np.array([p.lock_width for p in peaks], dtype=bool)

    def residual_jac(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        if theta.size != (n + (~lock_c).sum() + (~lock_w).sum()):
            raise ValueError("theta has wrong size")
        h = h0.copy()
        c = c0.copy()
        f = f0.copy()
        j = 0
        for i in range(n):
            h[i] = theta[j]
            j += 1
            if not lock_c[i]:
                c[i] = theta[j]
                j += 1
            if not lock_w[i]:
                f[i] = theta[j]
                j += 1
        if wmin_eval > 0.0:
            f = np.maximum(f, wmin_eval)

        pk_tuples = [(h[i], c[i], f[i], e[i]) for i in range(n)]
        unit_peaks = [(1.0, c[i], f[i], e[i]) for i in range(n)]
        comps_unit = perf.eval_components(x, unit_peaks)
        model = perf.eval_total(x, pk_tuples)
        cols = []
        for i in range(n):
            _, d_dc, d_df = pv_and_grads(x, h[i], c[i], f[i], e[i])
            cols.append(comps_unit[i])
            if not lock_c[i]:
                cols.append(d_dc)
            if not lock_w[i]:
                cols.append(d_df)
        J = np.column_stack(cols) if cols else np.zeros((x.size, 0))

        base = baseline if baseline is not None else 0.0
        if mode == "add":
            r = model + base - y
        elif mode == "subtract":
            r = model - (y - base)
        else:  # pragma: no cover - unknown mode
            raise ValueError("unknown mode")
        if w is not None:
            r = r * w
            J = J * w[:, None]
        return r, J

    return residual_jac


def _xp(arr: np.ndarray):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def jacobian_fd(residual_fn: Callable[[np.ndarray], np.ndarray],
                theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Finite-difference Jacobian for ``residual_fn``.

    Works with either NumPy or CuPy arrays depending on the type of ``theta``.
    """

    xp = _xp(theta)
    theta = xp.asarray(theta, dtype=float)
    r0 = residual_fn(theta)
    jac = xp.zeros((r0.size, theta.size), dtype=r0.dtype)
    for i in range(theta.size):
        step = xp.zeros_like(theta)
        step[i] = eps
        r1 = residual_fn(theta + step)
        jac[:, i] = (r1 - r0) / eps
    return jac
