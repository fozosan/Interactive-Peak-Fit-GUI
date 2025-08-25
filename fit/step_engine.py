"""Single-iteration fitting engine used by the Step ▶ feature."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core.peaks import Peak
from core.jacobians import pv_and_grads
from core.weights import noise_weights, robust_weights, combine_weights


def _theta_from_peaks(peaks: Sequence[Peak]) -> np.ndarray:
    """Flatten ``peaks`` into a parameter vector."""

    arr: list[float] = []
    for p in peaks:
        arr.extend([p.center, p.height, p.fwhm, p.eta])
    return np.asarray(arr, dtype=float)


def step_once(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    mode: str,
    baseline: np.ndarray | None,
    loss: str,
    weight_mode: str,
    damping: float,
    trust_radius: float,
    bounds: Sequence | None,
    wmin_eval: float = 0.0,
    f_scale: float = 1.0,
    max_backtracks: int = 10,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-12,
    min_step_ratio: float = 1e-9,
) -> tuple[np.ndarray, float, float, dict]:
    """Perform one weighted Gauss-Newton/LM step with backtracking.

    ``weight_mode`` selects the noise-weighting strategy and ``loss`` controls
    the robust IRLS weights.  Noise and robust weights are combined via
    :func:`core.weights.combine_weights` so that the behaviour matches the full
    solvers.  ``wmin_eval`` enforces a minimum width during model evaluation to
    guard against singularities. Armijo-style backtracking (step halving) is
    used to ensure that the returned step does not increase the weighted cost.

    Returns
    -------
    theta : ``np.ndarray``
        Flattened parameter vector after the step (or the original vector if
        the step was rejected).
    cost : float
        Weighted cost at ``theta``.
    cost0 : float
        Cost at the start of the step.
    info : dict
        Dictionary with ``step_norm``, ``accepted``, ``lambda``, ``backtracks``
        and ``reason`` for diagnostic use.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    n = len(peaks)
    c = np.array([p.center for p in peaks], dtype=float)
    h = np.array([p.height for p in peaks], dtype=float)
    f = np.array([p.fwhm for p in peaks], dtype=float)
    if wmin_eval > 0.0:
        f = np.maximum(f, wmin_eval)
    eta = np.array([p.eta for p in peaks], dtype=float)

    model = np.zeros_like(x)
    dcols = []
    for i in range(n):
        pv, d_dc, d_df = pv_and_grads(x, h[i], c[i], f[i], eta[i])
        model += pv
        base = pv / h[i] if h[i] != 0 else pv_and_grads(x, 1.0, c[i], f[i], eta[i])[0]
        dcols.extend([d_dc, base, d_df])
    J = np.column_stack(dcols) if dcols else np.zeros((x.size, 0))

    base_arr = baseline if baseline is not None else 0.0
    y_target = y - base_arr
    r = model - y_target

    w_noise = noise_weights(y_target, weight_mode)
    lam = float(damping)
    for _ in range(2):
        w_robust = robust_weights(r, loss, f_scale)
        w = combine_weights(w_noise, w_robust)
        if w is None:
            r_w = r
            J_w = J
        else:
            r_w = r * w
            J_w = J * w[:, None]
        if np.all(np.isfinite(r_w)) and np.all(np.isfinite(J_w)):
            break
        lam = lam * 10 if lam else 1e-3
    else:
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "nonfinite"}
        return _theta_from_peaks(peaks), float("inf"), float("inf"), info

    cost0 = 0.5 * float(r_w @ r_w)
    if not np.isfinite(cost0):
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "nonfinite"}
        return _theta_from_peaks(peaks), float(cost0), float(cost0), info

    JTJ = J_w.T @ J_w
    if lam:
        JTJ = JTJ + np.eye(JTJ.shape[0]) * lam
    rhs = -J_w.T @ r_w
    try:
        delta = np.linalg.solve(JTJ, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - singular matrix
        delta, *_ = np.linalg.lstsq(JTJ, rhs, rcond=None)

    if np.isfinite(trust_radius):
        norm = np.linalg.norm(delta)
        if norm > trust_radius and norm > 0:
            delta *= trust_radius / norm

    theta0 = _theta_from_peaks(peaks)
    norm_theta = np.linalg.norm(theta0)
    denom = max(1.0, norm_theta)
    if np.linalg.norm(delta) / denom < min_step_ratio:
        info = {"lambda": lam, "backtracks": 0, "step_norm": 0.0, "accepted": False, "reason": "tiny_step"}
        cost0 = float(cost0)
        return theta0, cost0, cost0, info
    mask = np.ones(theta0.size, dtype=bool)
    mask[3::4] = False  # do not update eta
    theta_base = theta0[mask]

    if bounds is not None:
        lb = np.asarray(bounds[0], dtype=float)[mask]
        ub = np.asarray(bounds[1], dtype=float)[mask]
    else:
        lb = ub = None

    step = delta.copy()
    accepted = False
    theta_reduced = theta_base.copy()
    cost = cost0
    n_bt = 0
    reason = "no_decrease"
    for _ in range(max_backtracks + 1):
        theta_try = theta_base + step
        if lb is not None and ub is not None:
            theta_try = np.minimum(np.maximum(theta_try, lb), ub)
        theta1 = theta0.copy()
        theta1[mask] = theta_try

        c_new = theta1[0::4]
        h_new = theta1[1::4]
        f_new = theta1[2::4]
        if wmin_eval > 0.0:
            f_new = np.maximum(f_new, wmin_eval)
        eta_new = theta1[3::4]
        model_new = np.zeros_like(x)
        for i in range(n):
            pv, _, _ = pv_and_grads(x, h_new[i], c_new[i], f_new[i], eta_new[i])
            model_new += pv
        r_new = model_new - y_target
        w_rob_new = robust_weights(r_new, loss, f_scale)
        w_new = combine_weights(w_noise, w_rob_new)
        if w_new is None:
            r_w_new = r_new
        else:
            r_w_new = r_new * w_new
        cost_new = 0.5 * float(r_w_new @ r_w_new)
        if np.isfinite(cost_new) and cost_new < cost0 - max(abs_tol, rel_tol * cost0):
            theta_reduced = theta_try
            cost = cost_new
            accepted = True
            reason = "accepted"
            break
        if not np.isfinite(cost_new):
            reason = "nonfinite"
            break
        if n_bt >= max_backtracks:
            reason = "no_decrease"
            break
        if lb is not None and ub is not None:
            hit = (theta_try <= lb + 1e-12) | (theta_try >= ub - 1e-12)
            step[hit] = 0.0
        step *= 0.5
        n_bt += 1

    theta_out = theta0.copy()
    if accepted:
        theta_out[mask] = theta_reduced
        step_norm = float(np.linalg.norm(theta_reduced - theta_base))
    else:
        step_norm = 0.0
        cost = cost0

    info = {
        "lambda": lam,
        "backtracks": n_bt,
        "step_norm": step_norm,
        "accepted": accepted,
        "reason": reason,
    }
    return theta_out, cost, cost0, info


def prepare_state(x, y, peaks, mode, baseline, opts):
    """Dispatch to solver-specific ``prepare_state``."""
    solver = opts.get("solver", "modern_vp")
    if solver == "classic":
        from . import classic
        return classic.prepare_state(x, y, peaks, mode, baseline, opts)
    if solver == "modern_trf":
        from . import modern
        return modern.prepare_state(x, y, peaks, mode, baseline, opts)
    if solver == "modern_vp":
        from . import modern_vp
        return modern_vp.prepare_state(x, y, peaks, mode, baseline, opts)
    if solver == "lmfit_vp":
        from . import lmfit_backend
        return lmfit_backend.prepare_state(x, y, peaks, mode, baseline, opts)
    raise ValueError(f"unknown solver '{solver}'")


def iterate(state):
    """Dispatch to solver-specific ``iterate``."""
    opts = state.get("options", {})
    solver = opts.get("solver", "modern_vp")
    if solver == "classic":
        from . import classic
        return classic.iterate(state)
    if solver == "modern_trf":
        from . import modern
        return modern.iterate(state)
    if solver == "modern_vp":
        from . import modern_vp
        return modern_vp.iterate(state)
    if solver == "lmfit_vp":
        from . import lmfit_backend
        return lmfit_backend.iterate(state)
    raise ValueError(f"unknown solver '{solver}'")
