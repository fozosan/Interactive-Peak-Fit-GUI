"""Residual and Jacobian construction for Peakfit 3.x."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
try:  # optional CuPy support
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy may be absent
    cp = None  # type: ignore

from .models import pv_sum
from .peaks import Peak



def build_residual(x: np.ndarray, y: np.ndarray, peaks: Sequence[Peak],
                   mode: str, baseline: np.ndarray | None,
                   loss: str, weights: np.ndarray | None) -> Callable[[np.ndarray], np.ndarray]:
    """Return a residual function ``r(theta)`` for the given data.

    ``theta`` is expected to contain ``4 * len(peaks)`` parameters ordered as
    ``(center, height, fwhm, eta)`` for each peak. Only a linear loss is
    currently supported. ``weights`` can provide per-point weighting.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    if loss != "linear":
        raise NotImplementedError("only linear loss supported")
    w = np.asarray(weights, dtype=float) if weights is not None else None

    def residual(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.size != 4 * len(peaks):
            raise ValueError("theta has wrong size")
        pk = []
        for i in range(len(peaks)):
            c, h, fw, eta = theta[4 * i : 4 * (i + 1)]
            pk.append(Peak(c, h, fw, eta))
        model = pv_sum(x, pk)
        if mode == "add":
            base = baseline if baseline is not None else 0.0
            r = model + base - y
        elif mode == "subtract":
            base = baseline if baseline is not None else 0.0
            r = model - (y - base)
        else:  # pragma: no cover - unknown mode
            raise ValueError("unknown mode")
        if w is not None:
            r = r * w
        return r

    return residual


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
