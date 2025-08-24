"""Single-iteration fitting engine used by the Step ▶ feature."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core.peaks import Peak
from core.jacobians import pv_and_grads
from core.robust import irls_weights


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
    weights: np.ndarray | None,
    damping: float,
    trust_radius: float,
    bounds: Sequence | None,
    f_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Perform one weighted Gauss-Newton/LM step.

    Parameters mirror those of the full solvers. ``weights`` should contain
    per-point noise weights (or ``None``). Robust losses are applied via IRLS
    using :func:`core.robust.irls_weights`.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    n = len(peaks)
    c = np.array([p.center for p in peaks], dtype=float)
    h = np.array([p.height for p in peaks], dtype=float)
    f = np.array([p.fwhm for p in peaks], dtype=float)
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
    if mode == "add":
        r = model + base_arr - y
    else:
        r = model - (y - base_arr)

    w = irls_weights(r, loss, f_scale)
    if weights is not None:
        w = w * weights
    r_w = r * w
    J_w = J * w[:, None]

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
    theta_reduced = theta0[mask] + delta
    if bounds is not None:
        lb = np.asarray(bounds[0], dtype=float)[mask]
        ub = np.asarray(bounds[1], dtype=float)[mask]
        theta_reduced = np.minimum(np.maximum(theta_reduced, lb), ub)
    theta1 = theta0.copy()
    theta1[mask] = theta_reduced

    # Recompute cost at the new parameters
    c_new = theta1[0::4]
    h_new = theta1[1::4]
    f_new = theta1[2::4]
    eta_new = theta1[3::4]
    model_new = np.zeros_like(x)
    for i in range(n):
        pv, _, _ = pv_and_grads(x, h_new[i], c_new[i], f_new[i], eta_new[i])
        model_new += pv
    if mode == "add":
        r_new = model_new + base_arr - y
    else:
        r_new = model_new - (y - base_arr)
    w_new = irls_weights(r_new, loss, f_scale)
    if weights is not None:
        w_new = w_new * weights
    cost = 0.5 * float((r_new * w_new) @ (r_new * w_new))
    return theta1, cost
