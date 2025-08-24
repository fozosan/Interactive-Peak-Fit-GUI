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
    max_backtracks: int = 8,
) -> tuple[np.ndarray, float, float, bool]:
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
    step_norm : float
        Euclidean norm of the accepted parameter update.
    accepted : bool
        ``True`` if the step decreased the cost, ``False`` otherwise.
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
    w_robust = robust_weights(r, loss, f_scale)
    w = combine_weights(w_noise, w_robust)

    r_w = r * w
    J_w = J * w[:, None]
    cost0 = 0.5 * float(r_w @ r_w)

    JTJ = J_w.T @ J_w
    if damping:
        JTJ = JTJ + np.eye(JTJ.shape[0]) * damping
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
        r_w_new = r_new * w_new
        cost_new = 0.5 * float(r_w_new @ r_w_new)
        if cost_new <= cost0:
            theta_reduced = theta_try
            cost = cost_new
            accepted = True
            break
        step *= 0.5

    theta_out = theta0.copy()
    if accepted:
        theta_out[mask] = theta_reduced
        step_norm = float(np.linalg.norm(theta_reduced - theta_base))
    else:
        step_norm = 0.0
        cost = cost0

    return theta_out, cost, step_norm, accepted
