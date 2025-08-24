"""Classic solver backend using SciPy's least squares routines.

This implementation mirrors the behaviour of Peakfit 2.7 where the solver
optimises peak heights as well as any unlocked centers and widths.  It relies
on :func:`scipy.optimize.least_squares` with simple linear loss and without the
robust weighting/restart features provided by the modern solver.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.peaks import Peak
from core.residuals import build_residual
from .bounds import pack_theta_bounds


class SolveResult(TypedDict):
    """Return structure for solver results."""

    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    """Solve the non-linear least squares problem for classic fitting.

    Parameters
    ----------
    x, y : ndarray
        Data points.
    peaks : list[Peak]
        Initial peak guesses with lock flags.
    mode : str
        Baseline mode (``"add"`` or ``"subtract"``).
    baseline : ndarray | None
        Baseline array if applicable.
    options : dict
        Solver options supporting ``centers_in_window``, ``min_fwhm`` and
        ``maxfev``.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base_arr = np.asarray(baseline, dtype=float) if baseline is not None else None

    # Handle baseline according to mode
    y_target = y
    base_model = None
    if base_arr is not None:
        if mode == "subtract":
            y_target = y - base_arr
        else:  # add
            base_model = base_arr

    maxfev = int(options.get("maxfev", 20000))

    theta0, (lb, ub) = pack_theta_bounds(peaks, x, options)
    resid_fn = build_residual(x, y_target, peaks, base_model, "linear", None)

    res = least_squares(resid_fn, theta0, max_nfev=maxfev, bounds=(lb, ub))

    theta = np.minimum(np.maximum(res.x, lb), ub)
    cost = 0.5 * float(res.cost)
    jac = res.jac if res.success else None
    cov = None
    if jac is not None:
        try:
            JTJ_inv = np.linalg.pinv(jac.T @ jac)
            cov = JTJ_inv
        except np.linalg.LinAlgError:  # pragma: no cover - singular
            cov = None

    return SolveResult(
        ok=bool(res.success),
        theta=theta,
        message=res.message,
        cost=cost,
        jac=jac,
        cov=cov,
        meta={"nfev": res.nfev, "njev": getattr(res, "njev", None)},
    )

