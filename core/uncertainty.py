"""Uncertainty estimation helpers used by tests and the GUI.

This module provides lightweight implementations of three uncertainty
estimators: asymptotic (based on the Jacobian), residual bootstrap and a
Bayesian Monte Carlo approach. The implementations are intentionally
minimal – they cover only the behaviour exercised in the test-suite.  The
functions are pure and operate on plain ``numpy`` arrays making them easy to
reuse in both the single file and batch paths.

The goal of this module is not to be the most feature complete uncertainty
package but rather to offer a small and well tested core that higher level
layers can rely on.  Each estimator returns a dictionary with a common
subset of fields:

``param_mean``
    The best parameter vector found (or the mean across samples).
``param_std``
    One standard deviation for each parameter.
``band``
    Optional tuple ``(x, lo, hi)`` describing a prediction band.
``metadata``
    Diagnostic information useful for logging.

The module purposefully has no side effects and performs no logging – the
caller is expected to handle that if desired.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from core.residuals import jacobian_fd
from infra import performance

__all__ = [
    "NotAvailable",
    "asymptotic_ci",
    "bootstrap_ci",
    "bayesian_ci",
]


class NotAvailable(RuntimeError):
    """Raised when an optional uncertainty backend is not available."""


# ---------------------------------------------------------------------------
# Utilities

def _complex_step_jacobian(
    f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, h: float = 1e-30
) -> np.ndarray:
    """Return the Jacobian of ``f`` using the complex-step method.

    If ``f`` cannot operate on complex numbers a ``TypeError`` is raised so
    that callers can fall back to a finite difference approximation.
    """

    theta = np.asarray(theta, float)
    n = theta.size
    J = np.empty((f(theta).size, n), float)
    for j in range(n):
        step = np.zeros_like(theta, dtype=complex)
        step[j] = 1j * h
        try:
            y1 = f(theta.astype(complex) + step)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("complex-step not supported") from exc
        if not np.iscomplexobj(y1):
            raise TypeError("complex-step not supported")
        J[:, j] = np.imag(y1) / h
    return J


def _central_jacobian(
    f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, scale: float = 1e-6
) -> np.ndarray:
    theta = np.asarray(theta, float)
    n = theta.size
    f0 = f(theta)
    J = np.empty((f0.size, n), float)
    for j in range(n):
        step = scale * max(1.0, abs(theta[j]))
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += step
        tm[j] -= step
        J[:, j] = (f(tp) - f(tm)) / (2.0 * step)
    return J


# ---------------------------------------------------------------------------
# Asymptotic CIs

def asymptotic_ci(
    theta: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jac_wrt_theta: Optional[np.ndarray],
    ymodel_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 0.05,
    svd_rcond: float = 1e-10,
    grad_mode: str = "complex",
) -> dict:
    """Return asymptotic parameter and prediction uncertainty."""

    theta = np.asarray(theta, float)

    r = np.asarray(residual_fn(theta), float)
    J = jacobian_fd(residual_fn, theta) if jac_wrt_theta is None else np.asarray(jac_wrt_theta, float)

    m, n = J.shape
    dof = max(m - n, 1)
    rss = float(np.dot(r, r))
    s2 = rss / dof

    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_inv = np.array([1 / si if si > svd_rcond * s[0] else 0.0 for si in s])
    JTJ_inv = (Vt.T * (s_inv ** 2)) @ Vt
    cov = s2 * JTJ_inv
    param_std = np.sqrt(np.diag(cov))

    cond = float(s[0] / s[-1]) if s[-1] != 0 else float("inf")
    rank = int((s > svd_rcond * s[0]).sum())

    y0 = np.asarray(ymodel_fn(theta), float)
    if grad_mode == "complex":
        try:
            G = _complex_step_jacobian(ymodel_fn, theta)
            grad_mode_used = "complex"
        except TypeError:
            G = _central_jacobian(ymodel_fn, theta)
            grad_mode_used = "central"
    else:
        G = _central_jacobian(ymodel_fn, theta)
        grad_mode_used = "central"
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    band_std = np.sqrt(np.maximum(var, 0.0))
    z = 1.96 if alpha == 0.05 else float(
        np.abs(np.quantile(np.random.standard_normal(100000), 1 - alpha / 2))
    )
    lo = y0 - z * band_std
    hi = y0 + z * band_std
    x = np.arange(y0.size)

    meta = {"rss": rss, "dof": dof, "cond": cond, "rank": rank, "grad_mode": grad_mode_used}

    return {"param_mean": theta, "param_std": param_std, "band": (x, lo, hi), "metadata": meta}


# ---------------------------------------------------------------------------
# Bootstrap CIs

def bootstrap_ci(
    engine: Callable[..., dict],
    data: dict,
    seeds: Optional[Iterable[int]] = None,
    n: int = 200,
    band_percentiles: tuple[float, float] = (2.5, 97.5),
    workers: int = 1,
    seed_root: Optional[int] = None,
) -> dict:
    """Residual bootstrap uncertainty estimation."""

    base_res = engine(**data, return_jacobian=True)
    theta = np.asarray(base_res["theta"], float)
    resid_fn = base_res["residual_fn"]
    r = resid_fn(theta)
    x = np.asarray(data["x"], float)
    y = np.asarray(data["y"], float)
    mask = np.asarray(data["fit_mask"], bool)

    y_model = y[mask] + r

    rng = np.random.default_rng(seed_root)
    samples: list[np.ndarray] = []
    curves: list[np.ndarray] = []

    for i in range(int(n)):
        idx = rng.integers(0, r.size, r.size)
        r_star = r[idx]
        y_boot_mask = y_model - r_star
        y_boot = y.copy()
        y_boot[mask] = y_boot_mask
        boot_data = dict(data)
        boot_data["y"] = y_boot
        res_i = engine(**boot_data)
        samples.append(np.asarray(res_i["theta"], float))
        if band_percentiles is not None:
            curves.append(base_res["ymodel_fn"](samples[-1]))

    samples_arr = np.vstack(samples) if samples else np.empty((0, theta.size))
    param_mean = samples_arr.mean(axis=0) if samples_arr.size else theta
    param_std = samples_arr.std(axis=0, ddof=1) if samples_arr.shape[0] > 1 else np.zeros_like(theta)
    ci = (
        np.percentile(samples_arr, [2.5, 97.5], axis=0)
        if samples_arr.size
        else np.tile(theta, (2, 1))
    )

    band = None
    if curves:
        curves_arr = np.vstack(curves)
        lo = np.percentile(curves_arr, band_percentiles[0], axis=0)
        hi = np.percentile(curves_arr, band_percentiles[1], axis=0)
        band = (x, lo, hi)

    return {
        "param_mean": param_mean,
        "param_std": param_std,
        "param_ci": ci,
        "band": band,
        "metadata": {"n": n},
        "samples": samples_arr,
    }


# ---------------------------------------------------------------------------
# Bayesian CIs

def bayesian_ci(
    engine: Callable[..., dict],
    data: dict,
    walkers: int = 32,
    steps: int = 1000,
    burn: int = 300,
    thin: int = 2,
    band: bool = True,
) -> dict:
    """Bayesian uncertainty estimation using ``emcee``."""

    try:  # pragma: no cover - optional dependency
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise NotAvailable("emcee not installed") from exc

    base_res = engine(**data, return_jacobian=True)
    theta0 = np.asarray(base_res["theta"], float)
    resid_fn = base_res["residual_fn"]
    bounds = base_res.get("bounds")
    r = resid_fn(theta0)
    dof = max(r.size - theta0.size, 1)
    sigma2 = float(np.dot(r, r)) / dof

    ndim = theta0.size
    nwalkers = max(int(walkers), 2 * ndim)
    rng = np.random.default_rng()
    p0 = theta0 + 1e-4 * rng.standard_normal((nwalkers, ndim))

    def log_prob(theta: np.ndarray) -> float:
        if bounds is not None:
            lo, hi = bounds
            if np.any(theta < lo) or np.any(theta > hi):
                return -np.inf
        rr = resid_fn(theta)
        return -0.5 * np.dot(rr, rr) / sigma2

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, int(steps), progress=False)
    chain = sampler.get_chain(discard=int(burn), thin=int(thin), flat=True)
    if chain.size == 0:
        chain = theta0[None, :]

    param_mean = chain.mean(axis=0)
    param_std = chain.std(axis=0, ddof=1) if chain.shape[0] > 1 else np.zeros(ndim)
    ci = np.percentile(chain, [2.5, 97.5], axis=0)

    pred_band = None
    if band:
        curves = np.vstack([base_res["ymodel_fn"](th) for th in chain])
        lo = np.percentile(curves, 2.5, axis=0)
        hi = np.percentile(curves, 97.5, axis=0)
        pred_band = (np.asarray(data["x"], float), lo, hi)

    return {
        "param_mean": param_mean,
        "param_std": param_std,
        "param_ci": ci,
        "band": pred_band,
        "metadata": {"n": chain.shape[0], "accept": float(np.mean(sampler.acceptance_fraction))},
        "samples": chain,
    }
