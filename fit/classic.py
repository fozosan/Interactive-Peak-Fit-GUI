"""Simplified Classic solver and stepper."""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
from scipy.optimize import least_squares

from core.peaks import Peak
from core.jacobians import pv_and_grads
from core.models import pv_sum
from core.weights import noise_weights, robust_weights, combine_weights
from core.bounds_classic import make_bounds_classic


class SolveResult(TypedDict):
    success: bool
    theta: np.ndarray
    peaks: list[Peak]
    cost: float
    rmse: float
    message: str
    meta: dict


def _model_jac(theta: np.ndarray, struct: list[dict], x: np.ndarray, fwhm_lo: float):
    model = np.zeros_like(x)
    J = np.zeros((x.size, len(theta)))
    for s in struct:
        h = max(theta[s["ih"]], 0.0)
        c = s["c_fixed"] if s["ic"] is None else theta[s["ic"]]
        w = s["w_fixed"] if s["iw"] is None else theta[s["iw"]]
        w = max(w, fwhm_lo)
        e = s["eta"]
        pv, d_dc, d_df = pv_and_grads(x, h, c, w, e)
        model += pv
        base = pv / h if h != 0 else pv_and_grads(x, 1.0, c, w, e)[0]
        J[:, s["ih"]] = base
        if s["ic"] is not None:
            J[:, s["ic"]] = d_dc
        if s["iw"] is not None:
            J[:, s["iw"]] = d_df
    return model, J


def _peaks_from_theta(theta: np.ndarray, struct: list[dict], fwhm_lo: float) -> list[Peak]:
    out: list[Peak] = []
    for s in struct:
        h = float(max(theta[s["ih"]], 0.0))
        c = float(s["c_fixed"] if s["ic"] is None else theta[s["ic"]])
        w = float(s["w_fixed"] if s["iw"] is None else theta[s["iw"]])
        w = max(w, fwhm_lo)
        out.append(Peak(c, h, w, float(s["eta"])) )
    return out


def _initial_state(x, y_target, peaks, opts):
    p0, bounds, struct = make_bounds_classic(x, y_target, peaks,
                                             fit_window=opts.get("fit_window"),
                                             fwhm_min=opts.get("fwhm_min"))
    dx = np.median(np.diff(np.sort(x))) if x.size > 1 else 1.0
    fwhm_lo = max(opts.get("fwhm_min", 0.0) or 0.0, 2.0 * dx, np.finfo(float).eps)
    w_noise = noise_weights(opts.get("weights", "none"), y_target)
    state = {
        "theta": p0.copy(),
        "bounds": bounds,
        "struct": struct,
        "x": np.asarray(x, float),
        "y": np.asarray(y_target, float),
        "w_noise": w_noise,
        "loss": opts.get("loss", "linear"),
        "f_scale": opts.get("f_scale", 1.0),
        "lambda": opts.get("lambda", 0.0),
        "trust_radius": opts.get("trust_radius", np.inf),
        "max_backtracks": opts.get("max_backtracks", 8),
        "fwhm_lo": fwhm_lo,
    }
    state["peaks"] = _peaks_from_theta(p0, struct, fwhm_lo)
    return state


def solve(x, y, peaks, mode, baseline, opts) -> SolveResult:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    if mode == "add":
        y_target = y
    else:
        b = base if base is not None else 0.0
        y_target = y - b
    opts = opts or {}
    state = _initial_state(x, y_target, peaks, opts)
    p0 = state["theta"]
    bounds = state["bounds"]
    struct = state["struct"]
    fwhm_lo = state["fwhm_lo"]
    w_noise = state["w_noise"]
    loss = state["loss"]
    f_scale = state["f_scale"]

    def fun(t):
        model, _ = _model_jac(t, struct, x, fwhm_lo)
        r = model - y_target
        w_rob = robust_weights(loss, r, f_scale)
        w = combine_weights(w_noise, w_rob)
        if w is None:
            return r
        return r * w

    def jac(t):
        model, J = _model_jac(t, struct, x, fwhm_lo)
        r = model - y_target
        w_rob = robust_weights(loss, r, f_scale)
        w = combine_weights(w_noise, w_rob)
        if w is None:
            return J
        return J * w[:, None]

    res = least_squares(fun, p0, jac=jac, bounds=bounds, method="trf",
                        max_nfev=int(opts.get("maxfev", 20000)))
    theta = np.minimum(np.maximum(res.x, bounds[0]), bounds[1])
    peaks_out = _peaks_from_theta(theta, struct, fwhm_lo)
    model_final = pv_sum(x, peaks_out)
    rmse = float(np.sqrt(np.mean((model_final - y_target) ** 2)))
    theta_full = np.empty(4 * len(peaks_out))
    for i, pk in enumerate(peaks_out):
        theta_full[4*i:4*(i+1)] = [pk.center, pk.height, pk.fwhm, pk.eta]
    return SolveResult(
        success=bool(res.success),
        theta=theta_full,
        peaks=peaks_out,
        cost=float(res.cost),
        rmse=rmse,
        message=res.message,
        meta={"nfev": res.nfev},
    )


def prepare_state(x, y, peaks, mode, baseline, opts):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    if mode == "add":
        y_target = y
    else:
        b = base if base is not None else 0.0
        y_target = y - b
    opts = opts or {}
    state = _initial_state(x, y_target, peaks, opts)
    return {"state": state}


def iterate(state):
    if "theta" not in state or "struct" not in state:
        init = _initial_state(state["x_fit"], state["y_fit"], state["peaks"], state.get("options", {}))
        state.update(init)

    theta0 = state["theta"].copy()
    struct = state["struct"]
    x = state.get("x", state.get("x_fit"))
    y = state.get("y", state.get("y_fit"))
    fwhm_lo = state["fwhm_lo"]
    w_noise = state.get("w_noise")
    loss = state.get("loss", "linear")
    f_scale = state.get("f_scale", 1.0)
    lam = state.get("lambda", 0.0)
    trust = state.get("trust_radius", np.inf)
    max_bt = int(state.get("max_backtracks", 8))
    min_step_ratio = float(state.get("min_step_ratio", 1e-9))
    lo, hi = state["bounds"]

    model0, J0 = _model_jac(theta0, struct, x, fwhm_lo)
    r0 = model0 - y
    w_rob0 = robust_weights(loss, r0, f_scale)
    w0 = combine_weights(w_noise, w_rob0)
    if w0 is None:
        r_w0 = r0
        J_w0 = J0
    else:
        r_w0 = r0 * w0
        J_w0 = J0 * w0[:, None]
    cost0 = 0.5 * float(r_w0 @ r_w0)
    if not np.isfinite(cost0):
        state["accepted"] = False
        state["step_norm"] = 0.0
        return state, False, float(cost0), float(cost0), {"backtracks": 0, "step_norm": 0.0, "lambda": lam, "reason": "nan_guard"}

    JTJ = J_w0.T @ J_w0
    if lam:
        JTJ = JTJ + np.eye(JTJ.shape[0]) * lam
    rhs = -J_w0.T @ r_w0
    try:
        delta = np.linalg.solve(JTJ, rhs)
    except np.linalg.LinAlgError:
        delta, *_ = np.linalg.lstsq(JTJ, rhs, rcond=None)

    if np.isfinite(trust):
        nrm = np.linalg.norm(delta)
        if nrm > trust and nrm > 0:
            delta *= trust / nrm

    if np.linalg.norm(delta) / max(1.0, np.linalg.norm(theta0)) < min_step_ratio:
        state["accepted"] = False
        state["step_norm"] = 0.0
        return state, False, cost0, cost0, {"backtracks": 0, "step_norm": 0.0, "lambda": lam, "reason": "tiny_step"}

    theta_try = theta0 + delta
    theta_try = np.minimum(np.maximum(theta_try, lo), hi)
    step = theta_try - theta0
    accepted = False
    cost1 = cost0
    n_bt = 0
    reason = "no_decrease"
    for _ in range(max_bt + 1):
        model1, _ = _model_jac(theta0 + step, struct, x, fwhm_lo)
        r1 = model1 - y
        w_rob1 = robust_weights(loss, r1, f_scale)
        w1 = combine_weights(w_noise, w_rob1)
        if w1 is None:
            r_w1 = r1
        else:
            r_w1 = r1 * w1
        cost1 = 0.5 * float(r_w1 @ r_w1)
        if np.isfinite(cost1) and cost1 < cost0 - max(1e-12, 1e-6 * cost0):
            theta_try = np.minimum(np.maximum(theta0 + step, lo), hi)
            accepted = True
            reason = "accepted"
            break
        if not np.isfinite(cost1):
            reason = "nonfinite"
            break
        if n_bt >= max_bt:
            reason = "no_decrease"
            break
        hit = (theta0 + step <= lo + 1e-12) | (theta0 + step >= hi - 1e-12)
        step[hit] = 0.0
        step *= 0.5
        n_bt += 1

    if accepted:
        theta_new = theta_try
        step_norm = float(np.linalg.norm(theta_new - theta0))
    else:
        theta_new = theta0
        step_norm = 0.0
        cost1 = cost0

    state["theta"] = theta_new
    state["cost"] = cost1
    state["accepted"] = accepted
    state["step_norm"] = step_norm
    state["peaks"] = _peaks_from_theta(theta_new, struct, fwhm_lo)
    info = {"backtracks": n_bt, "step_norm": step_norm, "lambda": lam, "reason": reason}
    return state, accepted, cost0, cost1, info
