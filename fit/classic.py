"""Minimal Classic solver and stepper using plain least squares."""
from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
from scipy.optimize import curve_fit

from core.peaks import Peak
from core.models import pv_sum
from core.bounds_classic import make_bounds_classic


class SolveResult(TypedDict):
    success: bool
    theta: np.ndarray
    peaks: list[Peak]
    cost: float
    rmse: float
    message: str
    meta: dict


def _pack_free(peaks: Sequence[Peak], *, wmin_floor: float = 1e-9):
    """Pack free parameters into a vector with simple bounds."""
    theta = []
    lo = []
    hi = []
    struct: list[dict] = []
    for pk in peaks:
        s: dict = {}
        s["ih"] = len(theta)
        theta.append(max(pk.height, 0.0))
        lo.append(0.0)
        hi.append(np.inf)
        if pk.lock_center:
            s["ic"] = None
            s["c_fixed"] = pk.center
        else:
            s["ic"] = len(theta)
            theta.append(pk.center)
            lo.append(-np.inf)
            hi.append(np.inf)
        if pk.lock_width:
            s["iw"] = None
            s["w_fixed"] = pk.fwhm
        else:
            s["iw"] = len(theta)
            theta.append(max(pk.fwhm, wmin_floor))
            lo.append(wmin_floor)
            hi.append(np.inf)
        s["eta"] = float(pk.eta)
        struct.append(s)
    return np.asarray(theta, float), (np.asarray(lo, float), np.asarray(hi, float)), struct


def _theta_full(theta: np.ndarray, struct: Sequence[dict], wmin_eval: float) -> np.ndarray:
    out = []
    for s in struct:
        h = max(theta[s["ih"]], 0.0)
        c = s["c_fixed"] if s["ic"] is None else theta[s["ic"]]
        w = s["w_fixed"] if s["iw"] is None else theta[s["iw"]]
        w = max(w, wmin_eval)
        out.extend([c, h, w, s["eta"]])
    return np.asarray(out, float)


def _theta_to_peaks(theta: np.ndarray, struct: Sequence[dict], wmin_eval: float) -> list[Peak]:
    out: list[Peak] = []
    for s in struct:
        h = float(max(theta[s["ih"]], 0.0))
        c = float(s["c_fixed"] if s["ic"] is None else theta[s["ic"]])
        w = float(s["w_fixed"] if s["iw"] is None else theta[s["iw"]])
        w = max(w, wmin_eval)
        out.append(Peak(c, h, w, float(s["eta"])) )
    return out


def _residual_builder(x_fit, y_fit, base_fit, struct, *, mode: str, wmin_eval: float):
    def residual(theta_free: np.ndarray) -> np.ndarray:
        peaks = []
        for s in struct:
            h = max(theta_free[s["ih"]], 0.0)
            c = s["c_fixed"] if s["ic"] is None else theta_free[s["ic"]]
            w = s["w_fixed"] if s["iw"] is None else theta_free[s["iw"]]
            w = max(w, wmin_eval)
            peaks.append(Peak(c, h, w, s["eta"]))
        model = pv_sum(x_fit, peaks)
        if mode == "add" and base_fit is not None:
            model = model + base_fit
        return model - y_fit

    return residual


def solve(x, y, peaks, mode, baseline, opts) -> SolveResult:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    y_target = y if mode == "add" else y - (base if base is not None else 0.0)
    base_fit = base if mode == "add" else None
    theta0, _b, struct = _pack_free(peaks)
    lo, hi, wmin_eval = make_bounds_classic(
        x,
        y_target,
        peaks,
        centers_in_window=bool(opts.get("bound_centers_to_window", True)),
        fwhm_min_factor=float(opts.get("fwhm_min_factor", 2.0)),
        fwhm_max_factor=float(opts.get("fwhm_max_factor", 0.5)),
        height_max_factor=float(opts.get("height_factor", 1.0)),
        margin_frac=float(opts.get("margin_frac", 0.0)),
    )
    theta0 = np.minimum(np.maximum(theta0, lo), hi)
    resid = _residual_builder(x, y_target, base_fit, struct, mode=mode, wmin_eval=wmin_eval)

    def model_func(xdata, *t):
        theta = np.asarray(t, float)
        return resid(theta) + y_target

    maxfev = int(opts.get("maxfev", 2000))
    popt, _ = curve_fit(model_func, x, y_target, p0=theta0, bounds=(lo, hi), maxfev=maxfev)
    popt = np.minimum(np.maximum(popt, lo), hi)
    peaks_out = _theta_to_peaks(popt, struct, wmin_eval)
    resid_final = resid(popt)
    cost = 0.5 * float(resid_final @ resid_final)
    rmse = float(np.sqrt(np.mean(resid_final ** 2)))
    theta_full = _theta_full(popt, struct, wmin_eval)
    return SolveResult(
        success=True,
        theta=theta_full,
        peaks=peaks_out,
        cost=cost,
        rmse=rmse,
        message="",
        meta={"nfev": maxfev},
    )


def prepare_state(x, y, peaks, mode, baseline, opts):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    y_target = y if mode == "add" else y - (base if base is not None else 0.0)
    base_fit = base if mode == "add" else None
    theta0, _b, struct = _pack_free(peaks)
    lo, hi, wmin_eval = make_bounds_classic(
        x,
        y_target,
        peaks,
        centers_in_window=bool(opts.get("bound_centers_to_window", True)),
        fwhm_min_factor=float(opts.get("fwhm_min_factor", 2.0)),
        fwhm_max_factor=float(opts.get("fwhm_max_factor", 0.5)),
        height_max_factor=float(opts.get("height_factor", 1.0)),
        margin_frac=float(opts.get("margin_frac", 0.0)),
    )
    theta0 = np.minimum(np.maximum(theta0, lo), hi)
    resid = _residual_builder(x, y_target, base_fit, struct, mode=mode, wmin_eval=wmin_eval)
    theta_full = _theta_full(theta0, struct, wmin_eval)
    state = {
        "x_fit": x,
        "y_fit": y,
        "baseline": base,
        "mode": mode,
        "theta_free": theta0,
        "theta": theta_full,
        "bounds": (lo, hi),
        "struct": struct,
        "residual": resid,
        "wmin_eval": wmin_eval,
        "lambda": opts.get("lambda", 0.0),
        "max_backtracks": int(opts.get("max_backtracks", 10)),
        "options": opts or {},
        "peaks": _theta_to_peaks(theta0, struct, wmin_eval),
    }
    return {"state": state}


def iterate(state, lam=None, backtrack_max=10, min_step=1e-8):
    if "theta_free" not in state or "struct" not in state:
        init = prepare_state(
            state["x_fit"],
            state["y_fit"],
            state["peaks"],
            state.get("mode", "add"),
            state.get("baseline"),
            state.get("options", {}),
        )["state"]
        state.update(init)

    theta = state["theta_free"]
    lo, hi = state["bounds"]
    resid = state["residual"]
    struct = state["struct"]
    wmin_eval = state["wmin_eval"]
    lam = state.get("lambda", 0.0) if lam is None else lam
    max_bt = state.get("max_backtracks", backtrack_max)

    for _ in range(2):
        r0 = resid(theta)
        if np.all(np.isfinite(r0)):
            break
        lam = lam * 10 if lam else 1e-3
    else:
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "nonfinite"}
        state["lambda"] = lam
        state["accepted"] = False
        state["step_norm"] = 0.0
        return state, False, float("inf"), float("inf"), info

    cost0 = 0.5 * float(r0 @ r0)
    if not np.isfinite(cost0):
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "nonfinite"}
        state["lambda"] = lam
        state["accepted"] = False
        state["step_norm"] = 0.0
        return state, False, float(cost0), float(cost0), info

    eps = np.sqrt(np.finfo(float).eps)
    J = np.empty((r0.size, theta.size))
    for i in range(theta.size):
        t_eps = theta.copy()
        t_eps[i] += eps
        r_eps = resid(t_eps)
        J[:, i] = (r_eps - r0) / eps

    JTJ = J.T @ J
    if lam:
        JTJ = JTJ + np.eye(JTJ.shape[0]) * lam
    rhs = -J.T @ r0
    try:
        delta = np.linalg.solve(JTJ, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover
        delta, *_ = np.linalg.lstsq(JTJ, rhs, rcond=None)

    if np.linalg.norm(delta) < min_step:
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "tiny_step"}
        state["lambda"] = lam
        state["accepted"] = False
        state["step_norm"] = 0.0
        return state, False, cost0, cost0, info

    step = delta.copy()
    accepted = False
    cost1 = cost0
    n_bt = 0
    reason = "no_decrease"
    while True:
        theta_try = np.minimum(np.maximum(theta + step, lo), hi)
        r1 = resid(theta_try)
        if not np.all(np.isfinite(r1)):
            reason = "nonfinite"
            break
        cost1 = 0.5 * float(r1 @ r1)
        if cost1 < cost0 and np.linalg.norm(step) >= min_step:
            theta_new = theta_try
            accepted = True
            reason = "accepted"
            break
        n_bt += 1
        if n_bt >= max_bt:
            break
        step *= 0.5
        lam *= 2

    if accepted:
        theta = theta_new
    else:
        cost1 = cost0
    theta_full = _theta_full(theta, struct, wmin_eval)
    peaks = _theta_to_peaks(theta, struct, wmin_eval)
    step_norm = float(np.linalg.norm(step)) if accepted else 0.0
    state.update(
        {
            "theta_free": theta,
            "theta": theta_full,
            "peaks": peaks,
            "lambda": lam,
            "accepted": accepted,
            "step_norm": step_norm,
        }
    )
    info = {"lambda": lam, "backtracks": min(n_bt, max_bt), "step_norm": step_norm, "accepted": accepted, "reason": reason}
    return state, accepted, cost0, cost1, info
