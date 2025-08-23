"""Asymptotic uncertainty estimation based on the Jacobian."""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class UncReport(TypedDict):
    type: str
    params: dict
    curve_band: dict
    meta: dict


def asymptotic(report_from_solver: dict, residual_builder) -> UncReport:
    """Return a curvature-based uncertainty report.

    Parameters
    ----------
    report_from_solver:
        Mapping matching the :class:`SolveResult` structure produced by the
        solver backends.
    residual_builder:
        Callable ``r(theta)`` returning residuals for the current problem.
    """

    theta = np.asarray(report_from_solver["theta"], dtype=float)
    residual_fn = residual_builder

    r = residual_fn(theta)
    jac = report_from_solver.get("jac")
    if jac is None:
        from core.residuals import jacobian_fd  # lazy import to avoid cycle

        jac = jacobian_fd(residual_fn, theta)

    # Covariance via (J^T J)^{-1} scaled by residual variance
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:  # pragma: no cover - singular matrix
        cov = np.linalg.pinv(jtj)

    dof = max(r.size - theta.size, 1)
    sigma2 = float(np.dot(r, r)) / dof
    cov = cov * sigma2
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)

    params = {"theta": theta, "sigma": sigma, "cov": cov, "corr": corr}

    return UncReport(
        type="asymptotic",
        params=params,
        curve_band={},
        meta={},
    )
