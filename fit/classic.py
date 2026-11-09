"""Simple SciPy curve_fit backend matching legacy v2.7 behaviour."""
from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
from scipy.optimize import curve_fit

from core.peaks import Peak
from core.weights import noise_weights
from core.baseline_model import joint_from_options
from infra import performance as perf
from .bounds import height_cap, held_heights, pack_theta_bounds

W_MIN = 1e-6  # minimal physical FWHM


def _pack(
    peaks: Sequence[Peak],
    x: np.ndarray,
    opts: dict,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[dict]]:
    """Flatten the free parameters, with the bounds every solver fits within.

    The bounds are fit.bounds.pack_theta_bounds's for these options, as for the
    Modern and LMFIT solvers, and the ones run_fit_consistent reports: the Classic
    solver once kept its own (FWHM unbounded above), and its result was clipped
    after the solve, onto a bound it had never seen.
    """
    _, (lo_full, hi_full) = pack_theta_bounds(peaks, x, opts)
    held = held_heights((lo_full, hi_full))
    theta: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    struct: list[dict] = []
    for i, pk in enumerate(peaks):
        c_lo, h_lo, w_lo = lo_full[4 * i : 4 * i + 3]
        c_hi, h_hi, w_hi = hi_full[4 * i : 4 * i + 3]
        s: dict = {"eta": float(pk.eta)}
        # height: free, unless its bounds hold it (fit.bounds.held_heights). A
        # held one is no parameter: a finite-difference step would not fit in it.
        if np.isfinite(held[i]):
            s["ih"] = None
            s["h_fixed"] = float(held[i])
        else:
            s["ih"] = len(theta)
            theta.append(max(pk.height, 0.0))
            lo.append(h_lo)
            hi.append(h_hi)
        # centre
        if pk.lock_center:
            s["ic"] = None
            s["c_fixed"] = float(pk.center)
        else:
            s["ic"] = len(theta)
            s["c_lo"], s["c_hi"] = float(c_lo), float(c_hi)
            theta.append(float(pk.center))
            lo.append(c_lo)
            hi.append(c_hi)
        # width
        if pk.lock_width:
            s["iw"] = None
            s["w_fixed"] = max(pk.fwhm, W_MIN)
        else:
            s["iw"] = len(theta)
            theta.append(max(pk.fwhm, W_MIN))
            lo.append(w_lo)
            hi.append(w_hi)
        struct.append(s)
    return np.asarray(theta, float), (np.asarray(lo, float), np.asarray(hi, float)), struct


def _params(theta: np.ndarray, s: dict) -> tuple[float, float, float]:
    """One peak's centre, height and width, from the free parameters ``theta``."""

    h = s["h_fixed"] if s["ih"] is None else max(theta[s["ih"]], 0.0)
    c = s["c_fixed"] if s["ic"] is None else theta[s["ic"]]
    c = float(np.clip(c, s.get("c_lo", -np.inf), s.get("c_hi", np.inf)))
    w = s["w_fixed"] if s["iw"] is None else theta[s["iw"]]
    return c, float(h), max(float(w), W_MIN)


def _theta_full(theta: np.ndarray, struct: Sequence[dict]) -> np.ndarray:
    out: list[float] = []
    for s in struct:
        c, h, w = _params(theta, s)
        out.extend([c, h, w, s["eta"]])
    return np.asarray(out, float)


def _theta_to_peaks(theta: np.ndarray, struct: Sequence[dict]) -> list[Peak]:
    out: list[Peak] = []
    for s in struct:
        c, h, w = _params(theta, s)
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
        pk_tuples = []
        for s in struct:
            c, h, w = _params(theta_free, s)
            pk_tuples.append((h, c, w, s["eta"]))
        model = perf.eval_total(x_fit, pk_tuples)
        if base_fit is not None:
            model = model + base_fit
        return model - y_target

    return residual


def solve(x, y, peaks, mode, baseline, opts):
    """Full curve_fit solution.

    Unweighted, unless ``opts["weights"]`` holds per-point weights (1 / sigma_i
    of a noise model): curve_fit then gets ``sigma = 1 / w``. With
    ``opts["baseline_basis"]`` (the joint baseline's columns, game plan 1.6) the
    coefficients are extra parameters after the peaks', unbounded, and the
    result carries them as ``baseline_coef``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None
    if mode == "add":
        y_target = y
        base_fit = base
    else:
        y_target = y - (base if base is not None else 0.0)
        base_fit = None
    weights = opts.get("weights")
    point_sigma = None
    if weights is not None and not isinstance(weights, str):
        w = noise_weights(y_target, weights)
        with np.errstate(divide="ignore"):
            point_sigma = np.where(w > 0, 1.0 / w, np.inf)
    opts = {**opts, "max_height": height_cap(y - (base if base is not None else 0.0), opts)}
    theta0, bounds, struct = _pack(peaks, x, opts)
    theta0 = np.clip(theta0, bounds[0], bounds[1])
    B = joint_from_options(opts, x.size)
    p_peaks = theta0.size
    if B is not None:
        # start the coefficients at their best fit to the data less the start peaks
        w_ls = np.ones_like(x) if point_sigma is None else np.where(np.isfinite(point_sigma), 1.0 / point_sigma, 0.0)
        rest = y_target - _build_residual(x, np.zeros_like(x), base_fit, struct)(theta0)
        c0 = np.linalg.lstsq(B * w_ls[:, None], rest * w_ls, rcond=None)[0]
        theta0 = np.r_[theta0, c0]
        m = B.shape[1]
        bounds = (np.r_[bounds[0], np.full(m, -np.inf)], np.r_[bounds[1], np.full(m, np.inf)])

    def model_func(xdata, *t):
        theta = np.asarray(t, float)
        pk_tuples = []
        for s in struct:
            c, h, w = _params(theta, s)
            pk_tuples.append((h, c, w, s["eta"]))
        model = perf.eval_total(xdata, pk_tuples)
        if base_fit is not None:
            model = model + base_fit
        if B is not None:
            model = model + B @ theta[p_peaks:]
        return model

    maxfev = int(opts.get("maxfev", 20000))
    popt, _ = curve_fit(
        model_func, x, y_target, p0=theta0, bounds=bounds, maxfev=maxfev, sigma=point_sigma
    )
    popt = np.clip(popt, bounds[0], bounds[1])
    coef = None
    if B is not None:
        coef = np.asarray(popt[p_peaks:], float)
        popt = popt[:p_peaks]
        base_fit = B @ coef if base_fit is None else base_fit + B @ coef
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
    if coef is not None:
        y_fit = y_fit + B @ coef
    return {
        "baseline_coef": coef,
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
    opts = {**opts, "max_height": height_cap(y - (base if base is not None else 0.0), opts)}
    theta0, bounds, struct = _pack(peaks, x, opts)
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

