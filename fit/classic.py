"""Simple SciPy curve_fit backend matching legacy v2.7 behaviour."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

from core.peaks import Peak
from infra import performance as perf

W_MIN = 1e-6  # minimal physical FWHM


def _pack(
    peaks: Sequence[Peak],
    *,
    centers_in_window: bool,
    xmin: float,
    xmax: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[dict]]:
    """Flatten free parameters and construct simple bounds."""
    theta: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    struct: list[dict] = []
    for pk in peaks:
        s: dict = {"eta": float(pk.eta)}
        # height always free
        s["ih"] = len(theta)
        theta.append(max(pk.height, 0.0))
        lo.append(0.0)
        hi.append(np.inf)
        # centre
        if pk.lock_center:
            s["ic"] = None
            c0 = pk.center
            if centers_in_window:
                c0 = float(np.clip(c0, xmin, xmax))
                s["c_lo"] = xmin
                s["c_hi"] = xmax
            s["c_fixed"] = c0
        else:
            s["ic"] = len(theta)
            c0 = pk.center
            if centers_in_window:
                c0 = float(np.clip(c0, xmin, xmax))
                lo.append(xmin)
                hi.append(xmax)
                s["c_lo"] = xmin
                s["c_hi"] = xmax
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
            theta.append(c0)
        # width
        if pk.lock_width:
            s["iw"] = None
            s["w_fixed"] = max(pk.fwhm, W_MIN)
        else:
            s["iw"] = len(theta)
            theta.append(max(pk.fwhm, W_MIN))
            lo.append(W_MIN)
            hi.append(np.inf)
        struct.append(s)
    return np.asarray(theta, float), (np.asarray(lo, float), np.asarray(hi, float)), struct


def _theta_full(theta: np.ndarray, struct: Sequence[dict]) -> np.ndarray:
    out: list[float] = []
    for s in struct:
        h = max(theta[s["ih"]], 0.0)
        c = s["c_fixed"] if s["ic"] is None else theta[s["ic"]]
        c = float(np.clip(c, s.get("c_lo", -np.inf), s.get("c_hi", np.inf)))
        w = s["w_fixed"] if s["iw"] is None else theta[s["iw"]]
        w = max(w, W_MIN)
        out.extend([c, h, w, s["eta"]])
    return np.asarray(out, float)


def _theta_to_peaks(theta: np.ndarray, struct: Sequence[dict]) -> list[Peak]:
    out: list[Peak] = []
    for s in struct:
        h = float(max(theta[s["ih"]], 0.0))
        c = float(s["c_fixed"] if s["ic"] is None else theta[s["ic"]])
        c = float(np.clip(c, s.get("c_lo", -np.inf), s.get("c_hi", np.inf)))
        w = float(s["w_fixed"] if s["iw"] is None else theta[s["iw"]])
        w = max(w, W_MIN)
        out.append(
            Peak(
                c,
                h,
                w,
                float(s["eta"]),
                s.get("ic") is None,
                s.get("iw") is None,
            )
        )
    return out


def _build_residual(
    x_fit: np.ndarray,
    y_target: np.ndarray,
    base_fit: np.ndarray | None,
    struct: Sequence[dict],
):
    def residual(theta_free: np.ndarray) -> np.ndarray:
        peaks = []
        for s in struct:
            h = max(theta_free[s["ih"]], 0.0)
            c = s["c_fixed"] if s["ic"] is None else theta_free[s["ic"]]
            c = float(np.clip(c, s.get("c_lo", -np.inf), s.get("c_hi", np.inf)))
            w = s["w_fixed"] if s["iw"] is None else theta_free[s["iw"]]
            w = max(w, W_MIN)
            peaks.append(Peak(c, h, w, s["eta"]))
        pk_tuples = [(p.height, p.center, p.fwhm, p.eta) for p in peaks]
        model = perf.eval_total(x_fit, pk_tuples)
        if base_fit is not None:
            model = model + base_fit
        return model - y_target

    return residual


def solve(x, y, peaks, mode, baseline, opts):
    """Full curve_fit solution using unweighted residuals."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    if mode == "add":
        y_target = y
        base_fit = base
    else:
        y_target = y - (base if base is not None else 0.0)
        base_fit = None
    centers_flag = bool(opts.get("centers_in_window", True))
    theta0, bounds, struct = _pack(peaks, centers_in_window=centers_flag, xmin=float(x.min()), xmax=float(x.max()))
    theta0 = np.clip(theta0, bounds[0], bounds[1])

    def model_func(xdata, *t):
        theta = np.asarray(t, float)
        peaks_loc = []
        for s in struct:
            h = max(theta[s["ih"]], 0.0)
            c = s["c_fixed"] if s["ic"] is None else theta[s["ic"]]
            c = float(np.clip(c, s.get("c_lo", -np.inf), s.get("c_hi", np.inf)))
            w = s["w_fixed"] if s["iw"] is None else theta[s["iw"]]
            w = max(w, W_MIN)
            peaks_loc.append(Peak(c, h, w, s["eta"]))
        pk_tuples = [(p.height, p.center, p.fwhm, p.eta) for p in peaks_loc]
        model = perf.eval_total(xdata, pk_tuples)
        if base_fit is not None:
            model = model + base_fit
        return model

    maxfev = int(opts.get("maxfev", 20000))
    popt, _ = curve_fit(model_func, x, y_target, p0=theta0, bounds=bounds, maxfev=maxfev)
    popt = np.clip(popt, bounds[0], bounds[1])
    resid_fun = _build_residual(x, y_target, base_fit, struct)
    resid_final = resid_fun(popt)
    cost = 0.5 * float(resid_final @ resid_final)
    rmse = float(np.sqrt(np.mean(resid_final**2)))
    peaks_out = _theta_to_peaks(popt, struct)
    theta_full = _theta_full(popt, struct)
    pk_tuples = [(p.height, p.center, p.fwhm, p.eta) for p in peaks_out]
    model_full = perf.eval_total(x, pk_tuples)
    if mode == "add" and base is not None:
        y_fit = model_full + base
    else:
        y_fit = model_full
    return {
        "success": True,
        "theta": theta_full,
        "peaks": peaks_out,
        "cost": cost,
        "rmse": rmse,
        "y_fit": y_fit,
        "info": {"nfev": maxfev},
        "meta": {"nfev": maxfev},
    }


def prepare_state(x, y, peaks, mode, baseline, opts):
    """Prepare iteration state for Step ▶."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    if mode == "add":
        y_target = y
        base_fit = base
    else:
        y_target = y - (base if base is not None else 0.0)
        base_fit = None
    centers_flag = bool(opts.get("centers_in_window", True))
    theta0, bounds, struct = _pack(peaks, centers_in_window=centers_flag, xmin=float(x.min()), xmax=float(x.max()))
    theta0 = np.clip(theta0, bounds[0], bounds[1])
    resid = _build_residual(x, y_target, base_fit, struct)
    r0 = resid(theta0)
    cost0 = 0.5 * float(r0 @ r0)
    theta_full = _theta_full(theta0, struct)
    state = {
        "x_fit": x,
        "y_target": y_target,
        "baseline": base,
        "mode": mode,
        "theta_free": theta0,
        "theta": theta_full,
        "bounds": bounds,
        "struct": struct,
        "residual": resid,
        "lambda": float(opts.get("lambda", 1.0)),
        "max_backtracks": int(opts.get("max_backtracks", 10)),
        "options": opts or {},
        "peaks": _theta_to_peaks(theta0, struct),
    }
    return {"state": state, "cost": cost0}


def iterate(state):
    """Perform a single damped Gauss–Newton step."""
    if "theta_free" not in state or "struct" not in state:
        init = prepare_state(
            state["x_fit"],
            state.get("y_fit", state.get("y_target")),
            state.get("peaks", []),
            state.get("mode", "add"),
            state.get("baseline"),
            state.get("options", {}),
        )["state"]
        state.update(init)

    theta = state["theta_free"]
    lo, hi = state["bounds"]
    resid = state["residual"]
    struct = state["struct"]
    lam = float(state.get("lambda", 1.0))
    max_bt = int(state.get("max_backtracks", 10))

    r0 = resid(theta)
    cost0 = 0.5 * float(r0 @ r0)
    eps = 1e-6 * np.maximum(1.0, np.abs(theta))

    attempt = 0
    accepted = False
    cost1 = cost0
    step_norm = 0.0
    reason = "no_decrease"
    n_bt = 0
    while attempt < 2:
        J = np.empty((r0.size, theta.size))
        good = True
        for i in range(theta.size):
            t_eps = theta.copy()
            t_eps[i] += eps[i]
            r_eps = resid(t_eps)
            if not np.all(np.isfinite(r_eps)):
                good = False
                break
            J[:, i] = (r_eps - r0) / eps[i]
        if not good or not np.all(np.isfinite(J)):
            good = False
            lam *= 10.0
            attempt += 1
            continue
        JTJ = J.T @ J + lam * np.eye(theta.size)
        rhs = -J.T @ r0
        try:
            delta = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover
            delta, *_ = np.linalg.lstsq(JTJ, rhs, rcond=None)
        step = delta.copy()
        n_bt = 0
        while True:
            theta_try = np.clip(theta + step, lo, hi)
            r1 = resid(theta_try)
            if not np.all(np.isfinite(r1)):
                lam *= 10.0
                attempt += 1
                good = False
                break
            cost1 = 0.5 * float(r1 @ r1)
            if cost1 < cost0:
                accepted = True
                theta = theta_try
                step_norm = float(np.linalg.norm(step))
                reason = "accepted"
                break
            n_bt += 1
            if n_bt >= max_bt:
                cost1 = cost0
                reason = "no_decrease"
                break
            step *= 0.5
        if good:
            break
        if attempt >= 2:
            break
    if attempt >= 2 and not accepted and not good:
        reason = "nonfinite"
        cost1 = cost0
        step_norm = 0.0
    theta_full = _theta_full(theta, struct)
    peaks = _theta_to_peaks(theta, struct)
    state.update(
        {
            "theta_free": theta,
            "theta": theta_full,
            "peaks": peaks,
            "lambda": lam,
        }
    )
    info = {
        "lambda": lam,
        "backtracks": n_bt,
        "step_norm": step_norm,
        "accepted": accepted,
        "reason": reason,
    }
    return state, accepted, cost0, cost1, info

