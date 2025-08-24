"""Modern solver with robust losses and analytic Jacobian."""

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

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    f_scale = options.get("f_scale", 1.0)
    maxfev = int(options.get("maxfev", 20000))
    restarts = int(options.get("restarts", 1))
    jitter_pct = float(options.get("jitter_pct", 0.0))

    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(np.abs(y), 1e-12, None)

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)

    all_locked = all(p.lock_center and p.lock_width for p in peaks)
    base = baseline if baseline is not None else 0.0
    y_target = y - base

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

    best = None
    best_cost = np.inf
    rng = np.random.default_rng(options.get("seed"))

    for _ in range(max(1, restarts)):
        if jitter_pct:
            jitter = 1.0 + jitter_pct / 100.0 * rng.standard_normal(theta0.shape)
            start = theta0 * jitter
            lb, ub = bounds
            start = np.minimum(np.maximum(start, lb), ub)
        else:
            start = theta0

        resid_jac = build_residual_jac(x, y, peaks, mode, baseline, weights)

        def fun(t):
            r, _ = resid_jac(t)
            return r

        def jac(t):
            _, J = resid_jac(t)
            return J

        res = least_squares(
            fun,
            start,
            jac=jac,
            method="trf",
            loss=loss,
            f_scale=f_scale,
            bounds=bounds,
            x_scale=x_scale,
            max_nfev=maxfev,
        )

        cost = float(res.cost)
        if cost < best_cost:
            best = res
            best_cost = cost

    if best is None:  # pragma: no cover - should not happen
        raise RuntimeError("least_squares failed")

    theta_full = theta0_full.copy()
    theta_full[indices] = best.x
    jac_full = best.jac
    cov = None
    if jac_full is not None:
        try:
            cov = np.linalg.pinv(jac_full.T @ jac_full)
        except np.linalg.LinAlgError:
            cov = None

    return SolveResult(
        ok=bool(best.success),
        theta=theta_full,
        message=best.message,
        cost=best_cost,
        jac=jac_full,
        cov=cov,
        meta={"nfev": best.nfev, "njev": getattr(best, "njev", None)},
    )

