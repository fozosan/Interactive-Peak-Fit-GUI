"""Modern solver with robust losses and analytic Jacobian."""

from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.residuals import build_residual_jac
from core.models import pv_design_matrix
from core.weights import noise_weights
from .bounds import pack_theta_bounds
from .utils import mad_sigma, robust_cost
from . import step_engine


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

    options = options.copy()
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weights = None if weight_mode == "none" else noise_weights(y_target, weight_mode)
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

    best = None
    best_cost = np.inf
    rng = np.random.default_rng(options.get("seed"))

    resid_jac = build_residual_jac(x, y, peaks, mode, baseline, weights)

    for _ in range(max(1, restarts)):
        if jitter_pct:
            jitter = 1.0 + jitter_pct / 100.0 * rng.standard_normal(theta0.shape)
            start = theta0 * jitter
            lb, ub = bounds
            start = np.minimum(np.maximum(start, lb), ub)
        else:
            start = theta0

        r0, J0 = resid_jac(start)
        local_f_scale = f_scale
        if loss != "linear" and (not np.isfinite(local_f_scale) or local_f_scale <= 0):
            sigma = mad_sigma(r0)
            local_f_scale = max(sigma, 1e-12)
        cost0 = robust_cost(r0, loss, local_f_scale)

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
            f_scale=local_f_scale,
            bounds=bounds,
            x_scale=x_scale,
            max_nfev=maxfev,
        )

        r_new, J_new = resid_jac(res.x)
        cost_new = robust_cost(r_new, loss, local_f_scale)
        backtracked = False
        if cost_new > cost0:
            theta_old = start
            step = res.x - start
            for alpha in [0.5, 0.25, 0.125, 0.0625]:
                theta_bt = theta_old + alpha * step
                r_bt, J_bt = resid_jac(theta_bt)
                cost_bt = robust_cost(r_bt, loss, local_f_scale)
                if cost_bt < cost0:
                    res.x = theta_bt
                    r_new = r_bt
                    J_new = J_bt
                    cost_new = cost_bt
                    backtracked = True
                    break
            else:
                res.x = start
                r_new, J_new = r0, J0
                cost_new = cost0
                backtracked = True
        if cost_new < best_cost:
            best = res
            best_cost = cost_new
            best.residual = r_new  # type: ignore
            best.jac = J_new
            best.f_scale = local_f_scale  # type: ignore
            best.backtracked = backtracked  # type: ignore

    if best is None:  # pragma: no cover - should not happen
        raise RuntimeError("least_squares failed")

    theta_full = theta0_full.copy()
    theta_full[indices] = best.x
    jac_full = getattr(best, "jac", None)
    cov = None
    if jac_full is not None and jac_full.size:
        try:
            cov = np.linalg.pinv(jac_full.T @ jac_full)
        except np.linalg.LinAlgError:
            cov = None

    sigma_final = mad_sigma(getattr(best, "residual", np.zeros(1)))
    meta = {
        "nfev": getattr(best, "nfev", None),
        "njev": getattr(best, "njev", None),
        "sigma": sigma_final,
        "f_scale": getattr(best, "f_scale", f_scale),
        "backtracked": getattr(best, "backtracked", False),
    }

    return SolveResult(
        ok=bool(best.success),
        theta=theta_full,
        message=best.message,
        cost=best_cost,
        jac=jac_full,
        cov=cov,
        meta=meta,
    )


def iterate(state: dict) -> dict:
    """Single iteration of the modern solver.

    This mirrors the behaviour of :func:`step_engine.step_once` but honours the
    solver options used by :func:`solve`.  Only a subset of the full solver
    functionality is required for stepping, so we reuse the generic step
    engine here.
    """

    x = state["x_fit"]
    y = state["y_fit"]
    peaks = state["peaks"]
    mode = state.get("mode", "subtract")
    baseline = state.get("baseline")
    options = state.get("options", {})

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")

    _, bounds = pack_theta_bounds(peaks, x, options)

    theta, cost, step_norm, accepted = step_engine.step_once(
        x,
        y,
        peaks,
        mode,
        baseline,
        loss=loss,
        weight_mode=weight_mode,
        damping=state.get("lambda", 0.0),
        trust_radius=state.get("trust_radius", np.inf),
        bounds=bounds,
        f_scale=options.get("f_scale", 1.0),
        max_backtracks=options.get("max_backtracks", 8),
    )

    state["theta"] = theta
    state["cost"] = cost
    state["step_norm"] = step_norm
    state["accepted"] = accepted
    return state

