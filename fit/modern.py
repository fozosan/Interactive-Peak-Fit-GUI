"""Modern solver backend employing Trust Region Reflective algorithm
with support for robust losses, weights and multi-start restarts."""
from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.residuals import build_residual
from .bounds import pack_theta_bounds


class SolveResult(TypedDict):
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
    """Solve the non-linear least squares problem using SciPy's TRF solver.

    Parameters in ``options`` follow the blueprint: ``loss`` (passed directly to
    :func:`scipy.optimize.least_squares`), ``weights`` (``none`` | ``poisson`` |
    ``inv_y``), ``f_scale``, ``maxfev``, ``restarts`` and ``jitter_pct``.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    f_scale = options.get("f_scale", 1.0)
    maxfev = int(options.get("maxfev", 20000))
    restarts = int(options.get("restarts", 1))
    jitter_pct = float(options.get("jitter_pct", 0.0))

    # construct weights
    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(y, 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(y, 1e-12, None)

    theta0, (lb, ub) = pack_theta_bounds(peaks, x, options)

    best = None
    best_cost = np.inf
    rng = np.random.default_rng(options.get("seed"))

    for _ in range(max(1, restarts)):
        if jitter_pct:
            jitter = 1.0 + jitter_pct / 100.0 * rng.standard_normal(theta0.shape)
            start = theta0 * jitter
            start = np.minimum(np.maximum(start, lb), ub)
        else:
            start = theta0

        resid_fn = build_residual(x, y, peaks, mode, baseline, loss, weights)

        res = least_squares(
            resid_fn,
            start,
            loss=loss,
            f_scale=f_scale,
            max_nfev=maxfev,
            bounds=(lb, ub),
        )

        cost = 0.5 * float(res.cost)
        if cost < best_cost:
            best = res
            best_cost = cost

    if best is None:  # pragma: no cover - should not happen
        raise RuntimeError("least_squares did not run")

    ok = bool(best.success)
    theta = best.x
    jac = best.jac if ok else None
    cov = None
    if jac is not None:
        try:
            JTJ_inv = np.linalg.pinv(jac.T @ jac)
            cov = JTJ_inv
        except np.linalg.LinAlgError:  # pragma: no cover - singular
            cov = None

    return SolveResult(
        ok=ok,
        theta=theta,
        message=best.message,
        cost=best_cost,
        jac=jac,
        cov=cov,
        meta={"nfev": best.nfev, "njev": getattr(best, "njev", None)},
    )
