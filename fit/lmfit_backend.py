"""LMFIT backend using variable projection."""
from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import nnls

from core.models import pv_design_matrix
from core.peaks import Peak
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


def _theta_from_peaks(peaks: Sequence[Peak]) -> np.ndarray:
    arr: list[float] = []
    for p in peaks:
        arr.extend([p.center, p.height, p.fwhm, p.eta])
    return np.asarray(arr, dtype=float)


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    try:
        import lmfit
    except Exception as exc:  # pragma: no cover - dependency missing
        return SolveResult(
            ok=False,
            theta=_theta_from_peaks(peaks),
            message=str(exc),
            cost=float("nan"),
            jac=None,
            cov=None,
            meta={},
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    f_scale_opt = float(options.get("f_scale", 0.0))
    lambda_c = float(options.get("lambda_c", 0.0))
    lambda_w = float(options.get("lambda_w", 0.0))

    options = options.copy()
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weights = noise_weights(y_target, weight_mode)
    p95 = float(np.percentile(np.abs(y_target), 95)) if y_target.size else 1.0
    max_height_factor = float(options.get("max_height_factor", np.inf))
    options["max_height"] = max_height_factor * p95
    options["max_fwhm"] = options.get("max_fwhm", 0.5 * (x.max() - x.min()))

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    min_fwhm = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)

    params = lmfit.Parameters()
    for i, p in enumerate(peaks):
        if not p.lock_center:
            params.add(
                f"c{i}", value=p.center, min=x.min(), max=x.max(), vary=True
            )
        else:
            params.add(f"c{i}", value=p.center, vary=False)
        if not p.lock_width:
            # Use per-parameter bounds derived by pack_theta_bounds (respects caps)
            w_lo = float(bounds_full[0][4 * i + 2])
            w_hi = float(bounds_full[1][4 * i + 2])
            params.add(
                f"w{i}", value=max(p.fwhm, min_fwhm), min=max(min_fwhm, w_lo), max=w_hi, vary=True
            )
        else:
            params.add(f"w{i}", value=max(p.fwhm, min_fwhm), vary=False)

    def residual(pars: lmfit.Parameters) -> np.ndarray:
        c = np.array([pars[f"c{i}"].value for i in range(len(peaks))], dtype=float)
        f = np.array([pars[f"w{i}"].value for i in range(len(peaks))], dtype=float)
        pk_tmp = [Peak(c[i], 1.0, f[i], peaks[i].eta) for i in range(len(peaks))]
        A = pv_design_matrix(x, pk_tmp)
        b = y_target
        if weights is not None:
            Aw = A * weights[:, None]
            bw = b * weights
        else:
            Aw = A
            bw = b
        h, _ = nnls(Aw, bw)
        h = np.minimum(h, options["max_height"])
        model = A @ h
        r = model - b
        if weights is not None:
            r = r * weights
        tether = []
        if lambda_c > 0:
            for i, p in enumerate(peaks):
                if not p.lock_center:
                    scale = np.sqrt(lambda_c) / max(p.fwhm, min_fwhm)
                    tether.append(scale * (c[i] - p.center))
        if lambda_w > 0:
            for i, p in enumerate(peaks):
                if not p.lock_width:
                    scale = np.sqrt(lambda_w)
                    tether.append(scale * np.log(f[i] / max(p.fwhm, min_fwhm)))
        if tether:
            r = np.concatenate([r, np.asarray(tether)])
        return r

    resid0 = residual(params)
    sigma = mad_sigma(resid0)
    f_scale = f_scale_opt if f_scale_opt > 0 else max(sigma, 1e-12)

    maxfev = int(options.get("maxfev", 20000))
    minimizer = lmfit.Minimizer(residual, params, nan_policy="omit")
    result = minimizer.minimize(
        method="least_squares",
        max_nfev=maxfev,
        loss=loss,
        f_scale=f_scale,
    )

    c = np.array([result.params[f"c{i}"].value for i in range(len(peaks))], dtype=float)
    f = np.array([result.params[f"w{i}"].value for i in range(len(peaks))], dtype=float)
    pk_final = [Peak(c[i], 1.0, f[i], peaks[i].eta) for i in range(len(peaks))]
    A = pv_design_matrix(x, pk_final)
    if weights is not None:
        Aw = A * weights[:, None]
        bw = y_target * weights
    else:
        Aw = A
        bw = y_target
    h, _ = nnls(Aw, bw)
    h = np.minimum(h, options["max_height"])
    model = A @ h
    r = model - y_target
    if weights is not None:
        r = r * weights
    cost = robust_cost(r, loss, f_scale)

    theta_full = theta0_full.copy()
    for i in range(len(peaks)):
        theta_full[4 * i + 0] = c[i]
        theta_full[4 * i + 1] = h[i]
        theta_full[4 * i + 2] = f[i]
    J = None

    return SolveResult(
        ok=result.success,
        theta=theta_full,
        message=result.message,
        cost=cost,
        jac=J,
        cov=None,
        meta={"nfev": result.nfev, "sigma": sigma, "f_scale": f_scale},
    )


def prepare_state(x, y, peaks, mode, baseline, opts):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    baseline_arr = np.asarray(baseline, float) if baseline is not None else None
    state = {
        "x_fit": x,
        "y_fit": y,
        "peaks": [Peak(p.center, p.height, p.fwhm, p.eta) for p in peaks],
        "mode": mode,
        "baseline": baseline_arr,
        "options": opts or {},
    }
    return {"state": state}


def iterate(state: dict):
    """Perform a single iteration using the lmfit backend."""

    x = state["x_fit"]
    y = state["y_fit"]
    peaks = state["peaks"]
    mode = state.get("mode", "subtract")
    baseline = state.get("baseline")
    options = state.get("options", {})

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")

    _, bounds = pack_theta_bounds(peaks, x, options)

    theta, cost1, cost0, info = step_engine.step_once(
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
        min_step_ratio=options.get("min_step_ratio", 1e-9),
    )

    state["theta"] = theta
    state["cost"] = cost1
    state["step_norm"] = info["step_norm"]
    state["accepted"] = info["accepted"]
    state["lambda"] = info["lambda"]
    state["peaks"] = [
        Peak(theta[4 * i], theta[4 * i + 1], theta[4 * i + 2], theta[4 * i + 3])
        for i in range(len(peaks))
    ]
    return state, info["accepted"], cost0, cost1, info
