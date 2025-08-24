"""Classic solver backend using SciPy's standard least-squares routine."""

from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import least_squares

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
    """Solve the full non-linear least squares problem with a linear loss.

    This function mirrors the behaviour of the classic 2.7 backend where all
    peak parameters (centre, height, FWHM and eta) are optimised simultaneously
    using SciPy's :func:`least_squares` with a standard (linear) loss.  It acts
    as the lightweight counterpart to :mod:`fit.modern` which adds robust losses
    and restarts.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    maxfev = int(options.get("maxfev", 20000))

    theta0, bounds = pack_theta_bounds(peaks, x, options)
    resid_fn = build_residual(x, y, peaks, mode, baseline, "linear", None)

    res = least_squares(
        resid_fn,
        theta0,
        method="trf",
        loss="linear",
        max_nfev=maxfev,
        bounds=bounds,
    )

    ok = bool(res.success)
    theta = res.x
    jac = res.jac if ok else None
    cov = None
    if jac is not None:
        try:
            cov = np.linalg.pinv(jac.T @ jac)
        except np.linalg.LinAlgError:  # pragma: no cover - singular
            cov = None

    return SolveResult(
        ok=ok,
        theta=theta,
        message=res.message,
        cost=float(res.cost),
        jac=jac,
        cov=cov,
        meta={"nfev": res.nfev, "njev": getattr(res, "njev", None)},
    )
