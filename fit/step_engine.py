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
    f_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Perform one Gauss-Newton/Levenberg-Marquardt step.

    ``loss`` can be ``linear``, ``soft_l1``, ``huber`` or ``cauchy`` and is
    implemented via an iteratively reweighted least squares (IRLS) scheme. Any
    ``weights`` provided are applied first, then the robust loss weights. Bounds
    are enforced by simple clipping after the step.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = (
        np.asarray(baseline, dtype=float) if baseline is not None else None
    )

    theta0 = _theta_from_peaks(peaks)
    lb = ub = None
    if bounds is not None:
        lb, ub = bounds
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        # ensure starting vector honours the bounds
        theta0 = np.minimum(np.maximum(theta0, lb), ub)

    resid_fn = build_residual(x, y, peaks, baseline, loss, weights)
    r0 = resid_fn(theta0)

    # robust loss via IRLS
    if loss != "linear":
        rs = r0 / f_scale
        if loss == "soft_l1":
            w = 1.0 / np.sqrt(1.0 + rs**2)
        elif loss == "huber":
            w = np.where(np.abs(rs) <= 1.0, 1.0, 1.0 / np.abs(rs))
        elif loss == "cauchy":
            w = 1.0 / (1.0 + rs**2)
        else:  # pragma: no cover - unknown loss
            raise ValueError("unknown loss")
        w_sqrt = np.sqrt(w)
        r0 = w_sqrt * r0
    else:
        w_sqrt = None

    J = jacobian_fd(resid_fn, theta0)
    if w_sqrt is not None:
        J = J * w_sqrt[:, None]

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
        theta1 = np.minimum(np.maximum(theta1, lb), ub)

    r1 = resid_fn(theta1)
    cost = 0.5 * float(r1 @ r1)
    return theta1, cost
