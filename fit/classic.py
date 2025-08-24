"""Classic solver backend using SciPy's least-squares with analytic Jacobian."""

from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.residuals import build_residual_jac
from core.models import pv_design_matrix
from .bounds import pack_theta_bounds


class SolveResult(TypedDict):
    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict


def _to_solver_vectors(theta0: np.ndarray, bounds, peaks, fwhm_min: float):
    lb, ub = bounds
    theta_list = []
    lb_list = []
    ub_list = []
    x_scale = []
    indices = []
    for i, p in enumerate(peaks):
        h = theta0[4 * i + 1]
        theta_list.append(h)
        lb_list.append(lb[4 * i + 1])
        ub_list.append(ub[4 * i + 1])
        x_scale.append(max(1.0, abs(h)))
        indices.append(4 * i + 1)
        if not p.lock_center:
            c = theta0[4 * i + 0]
            theta_list.append(c)
            lb_list.append(lb[4 * i + 0])
            ub_list.append(ub[4 * i + 0])
            x_scale.append(max(theta0[4 * i + 2], fwhm_min))
            indices.append(4 * i + 0)
        if not p.lock_width:
            w = theta0[4 * i + 2]
            theta_list.append(w)
            lb_list.append(lb[4 * i + 2])
            ub_list.append(ub[4 * i + 2])
            x_scale.append(max(w, fwhm_min))
            indices.append(4 * i + 2)
    return (
        np.asarray(theta_list, dtype=float),
        (np.asarray(lb_list, dtype=float), np.asarray(ub_list, dtype=float)),
        np.asarray(x_scale, dtype=float),
        np.asarray(indices, dtype=int),
    )


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    weight_mode = options.get("weights", "none")
    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(np.abs(y), 1e-12, None)

    options = options.copy()
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    p95 = float(np.percentile(np.abs(y_target), 95)) if y_target.size else 1.0
    max_height_factor = float(options.get("max_height_factor", np.inf))
    options["max_height"] = max_height_factor * p95
    options["max_fwhm"] = options.get("max_fwhm", 0.5 * (x.max() - x.min()))

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)

    all_locked = all(p.lock_center and p.lock_width for p in peaks)

    if all_locked:
        A = pv_design_matrix(x, peaks)
        if weights is not None:
            Aw = A * weights[:, None]
            yw = y_target * weights
        else:
            Aw = A
            yw = y_target
        h, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        model = A @ h
        r = model - y_target
        if weights is not None:
            r = r * weights
        cost = 0.5 * float(r @ r)
        theta_full = theta0_full.copy()
        for i, val in enumerate(h):
            theta_full[4 * i + 1] = val
        jac = Aw if weights is not None else A
        return SolveResult(
            ok=True,
            theta=theta_full,
            message="linear",
            cost=cost,
            jac=jac,
            cov=None,
            meta={"nfev": 1},
        )

    theta0, bounds, x_scale, indices = _to_solver_vectors(theta0_full, bounds_full, peaks, fwhm_min)
    resid_jac = build_residual_jac(x, y, peaks, mode, baseline, weights)

    def fun(t):
        r, _ = resid_jac(t)
        return r

    def jac(t):
        _, J = resid_jac(t)
        return J

    maxfev = int(options.get("maxfev", 20000))

    res = least_squares(
        fun,
        theta0,
        jac=jac,
        method="trf",
        loss="linear",
        bounds=bounds,
        x_scale=x_scale,
        max_nfev=maxfev,
    )

    ok = bool(res.success)
    theta_full = theta0_full.copy()
    theta_full[indices] = res.x
    jac_full = res.jac
    cov = None
    if jac_full is not None:
        try:
            cov = np.linalg.pinv(jac_full.T @ jac_full)
        except np.linalg.LinAlgError:
            cov = None

    return SolveResult(
        ok=ok,
        theta=theta_full,
        message=res.message,
        cost=float(res.cost),
        jac=jac_full,
        cov=cov,
        meta={"nfev": res.nfev, "njev": getattr(res, "njev", None)},
    )

