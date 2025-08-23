"""Single-iteration fitting engine used by the Step ▶ feature."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core.peaks import Peak
from core.residuals import build_residual, jacobian_fd


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
) -> tuple[np.ndarray, float]:
    """Perform one Gauss-Newton/Levenberg-Marquardt step.

    Returns the updated parameter vector and the new cost. Only a linear loss
    is currently supported; bounds are applied via simple clipping.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = (
        np.asarray(baseline, dtype=float) if baseline is not None else None
    )

    theta0 = _theta_from_peaks(peaks)
    resid_fn = build_residual(x, y, peaks, mode, baseline, loss, weights)
    r0 = resid_fn(theta0)
    J = jacobian_fd(resid_fn, theta0)
    JTJ = J.T @ J
    if damping:
        JTJ = JTJ + np.eye(JTJ.shape[0]) * damping
    rhs = -J.T @ r0
    try:
        delta = np.linalg.solve(JTJ, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - singular matrix
        delta, *_ = np.linalg.lstsq(JTJ, rhs, rcond=None)

    if np.isfinite(trust_radius):
        norm = np.linalg.norm(delta)
        if norm > trust_radius and norm > 0:
            delta *= trust_radius / norm

    theta1 = theta0 + delta
    if bounds is not None:
        lb, ub = bounds
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        theta1 = np.minimum(np.maximum(theta1, lb), ub)

    r1 = resid_fn(theta1)
    cost = 0.5 * float(r1 @ r1)
    return theta1, cost
