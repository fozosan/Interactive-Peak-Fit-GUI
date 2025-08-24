"""Classic solver backend using SciPy's least-squares with analytic Jacobian."""

from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.residuals import build_residual_jac
from core.models import pv_design_matrix
from core.weights import noise_weights
from core.bounds import make_bounds_classic
from . import step_engine


class SolveResult(TypedDict):
    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict
    hit_bounds: bool
    hit_mask: np.ndarray


def _to_solver_vectors(theta0: np.ndarray, bounds, peaks, wmin_eval: float):
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
            x_scale.append(max(theta0[4 * i + 2], wmin_eval))
            indices.append(4 * i + 0)
        if not p.lock_width:
            w = theta0[4 * i + 2]
            theta_list.append(w)
            lb_list.append(lb[4 * i + 2])
            ub_list.append(ub[4 * i + 2])
            x_scale.append(max(w, wmin_eval))
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
    cfg = {
        "bound_centers_to_window": options.get("bound_centers_to_window", True),
        "margin_frac": options.get("margin_frac", 0.0),
        "fwhm_min_factor": options.get("fwhm_min_factor", 2.0),
        "fwhm_max_factor": options.get("fwhm_max_factor", 0.5),
        "height_factor": options.get("height_factor", 3.0),
    }

    weights = None
    if weight_mode != "none":
        if mode == "add":
            y_target = y
        else:
            base = baseline if baseline is not None else 0.0
            y_target = y - base
        weights = noise_weights(y_target, weight_mode)

    (lb_full, ub_full), theta0_full = make_bounds_classic(
        x, y, peaks, None, mode, baseline, cfg
    )

    wmin_eval = float(lb_full[2]) if theta0_full.size >= 3 else 1e-6

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
            idx = 4 * i + 1
            theta_full[idx] = float(np.clip(val, lb_full[idx], ub_full[idx]))
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

    theta0, bounds, x_scale, indices = _to_solver_vectors(
        theta0_full, (lb_full, ub_full), peaks, wmin_eval
    )
    resid_jac = build_residual_jac(
        x, y, peaks, mode, baseline, weights, wmin_eval=wmin_eval
    )

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
    theta_full = np.minimum(np.maximum(theta_full, lb_full), ub_full)
    jac_full = res.jac
    cov = None
    if jac_full is not None:
        try:
            cov = np.linalg.pinv(jac_full.T @ jac_full)
        except np.linalg.LinAlgError:
            cov = None

    hit_mask = (np.isclose(theta_full, lb_full, atol=1e-12) |
                np.isclose(theta_full, ub_full, atol=1e-12))
    hit_bounds = bool(np.any(hit_mask))

    return SolveResult(
        ok=ok,
        theta=theta_full,
        message=res.message,
        cost=float(res.cost),
        jac=jac_full,
        cov=cov,
        meta={"nfev": res.nfev, "njev": getattr(res, "njev", None)},
        hit_bounds=hit_bounds,
        hit_mask=hit_mask,
    )


def iterate(state: dict) -> dict:
    """Perform a single iteration of the classic solver.

    The implementation is a thin wrapper around :func:`step_engine.step_once`
    so that the Step button in the GUI can advance the solution using the same
    residuals and bounds as the full solver.  ``state`` should contain the
    fields ``x_fit``, ``y_fit`` and ``peaks`` as well as optional ``baseline``
    and ``options`` dictionaries.
    """

    x = state["x_fit"]
    y = state["y_fit"]
    peaks = state["peaks"]
    mode = state.get("mode", "subtract")
    baseline = state.get("baseline")
    options = state.get("options", {})
    weight_mode = options.get("weights", "none")
    cfg = {
        "bound_centers_to_window": options.get("bound_centers_to_window", True),
        "margin_frac": options.get("margin_frac", 0.0),
        "fwhm_min_factor": options.get("fwhm_min_factor", 2.0),
        "fwhm_max_factor": options.get("fwhm_max_factor", 0.5),
        "height_factor": options.get("height_factor", 3.0),
    }
    (lb, ub), theta0 = make_bounds_classic(x, y, peaks, None, mode, baseline, cfg)
    for i, pk in enumerate(peaks):
        pk.center = theta0[4 * i + 0]
        pk.height = theta0[4 * i + 1]
        pk.fwhm = theta0[4 * i + 2]
        pk.eta = theta0[4 * i + 3]

    theta, cost, step_norm, accepted = step_engine.step_once(
        x,
        y,
        peaks,
        mode,
        baseline,
        loss="linear",
        weight_mode=weight_mode,
        damping=state.get("lambda", 0.0),
        trust_radius=state.get("trust_radius", np.inf),
        bounds=(lb, ub),
        wmin_eval=lb[2] if lb.size >= 3 else 1e-6,
        f_scale=options.get("f_scale", 1.0),
        max_backtracks=options.get("max_backtracks", 8),
    )

    hit_mask = (np.isclose(theta, lb, atol=1e-12) | np.isclose(theta, ub, atol=1e-12))
    state["theta"] = theta
    state["cost"] = cost
    state["step_norm"] = step_norm
    state["accepted"] = accepted
    state["hit_bounds"] = bool(np.any(hit_mask))
    state["hit_mask"] = hit_mask
    return state

