"""Classic solver backend using SciPy's least squares routines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from core.peaks import Peak
from core.residuals import build_residual
from .bounds import pack_theta_bounds


def _theta_from_peaks(peaks: Sequence[Peak]) -> np.ndarray:
    arr = []
    for p in peaks:
        arr.extend([p.center, p.height, p.fwhm, p.eta])
    return np.asarray(arr, dtype=float)

def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> dict:
    """Fit peak heights with centers/widths fixed using linear least squares.

    This lightweight implementation serves as an initial backend so the UI can
    demonstrate fitting. It solves for peak heights only and returns an array of
    full peak parameters to remain compatible with the blueprint API.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else 0.0

    if mode == "add":
        target = y - baseline
    elif mode == "subtract":
        target = y - baseline
    else:  # pragma: no cover - unknown mode
        return {
            "ok": False,
            "theta": _theta_from_peaks(peaks),
            "message": "unknown mode",
            "cost": float("nan"),
            "jac": None,
            "cov": None,
            "meta": {},
        }

    # enforce basic bounds on provided peak parameters
    x_min = float(x.min())
    x_max = float(x.max())
    min_fwhm = float(options.get("min_fwhm", 1e-6))
    clamp_center = bool(options.get("centers_in_window", False))
    clean: list[Peak] = []
    for p in peaks:
        c = p.center
        if clamp_center:
            c = float(np.clip(c, x_min, x_max))
        h = max(p.height, 0.0)
        w = max(p.fwhm, min_fwhm)
        e = float(np.clip(p.eta, 0.0, 1.0))
        clean.append(Peak(c, h, w, e))

    A_cols = []
    for p in clean:
        unit = Peak(p.center, 1.0, p.fwhm, p.eta)
        A_cols.append(pv_sum(x, [unit]))
    A = np.column_stack(A_cols) if A_cols else np.zeros((x.size, 0))
    try:
        heights, *_ = np.linalg.lstsq(A, target, rcond=None)
        heights = np.maximum(heights, 0.0)
        ok = True
        message = "linear least squares"
    except np.linalg.LinAlgError as exc:  # pragma: no cover - ill-conditioned
        heights = np.zeros(len(clean))
        ok = False
        message = str(exc)

    updated = [Peak(p.center, h, p.fwhm, p.eta) for p, h in zip(clean, heights)]
    theta = _theta_from_peaks(updated)
    model = pv_sum(x, updated)
    resid = target - model
    cost = float(0.5 * np.dot(resid, resid))

    return {
        "ok": ok,
        "theta": theta,
        "message": message,
        "cost": cost,
        "jac": None,
        "cov": None,
        "meta": {},
    }

