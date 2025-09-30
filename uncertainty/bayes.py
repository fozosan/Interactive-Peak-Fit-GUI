"""Bayesian uncertainty estimation via MCMC sampling."""
from __future__ import annotations

from typing import TypedDict

import numpy as np

from core.residuals import build_residual
from core.peaks import Peak
from infra import performance


class UncReport(TypedDict):
    type: str
    params: dict
    curve_band: dict
    meta: dict


def bayesian(
    priors: dict,
    like: str,
    init_from_solver: dict,
    sampler_cfg: dict,
    constraints: dict | None,
) -> UncReport:
    """Sample parameter posteriors using ``emcee``.

    This implementation supports a Gaussian likelihood with simple uniform
    bounds acting as weak priors. The input ``init_from_solver`` should provide
    ``x``, ``y``, ``peaks`` (template peaks), ``mode``, ``baseline`` and
    ``theta`` (solver result).
    """

    try:
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("emcee is required for Bayesian uncertainty") from exc

    cfg_perf = performance.get_parallel_config()
    performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)

    x = np.asarray(init_from_solver["x"], dtype=float)
    y = np.asarray(init_from_solver["y"], dtype=float)
    peaks_tpl: list[Peak] = list(init_from_solver["peaks"])
    mode = init_from_solver.get("mode", "add")
    baseline = init_from_solver.get("baseline")
    theta0 = np.asarray(init_from_solver["theta"], dtype=float)

    lb = np.asarray(priors.get("lb", -np.inf), dtype=float)
    ub = np.asarray(priors.get("ub", np.inf), dtype=float)
    sigma = float(priors.get("sigma", 1.0))

    resid_fn = build_residual(x, y, peaks_tpl, mode, baseline, "linear", None)

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta < lb) or np.any(theta > ub):
            return -np.inf
        r = resid_fn(theta)
        return -0.5 * float(np.dot(r, r) / (sigma**2))

    ndim = theta0.size
    nwalkers = int(sampler_cfg.get("nwalkers", 2 * ndim))
    nsteps = int(sampler_cfg.get("nsteps", 1000))
    seed_cfg = sampler_cfg.get("seed")
    if seed_cfg is None and cfg_perf.seed_all:
        seed_cfg = cfg_perf.seed_value
    rng = np.random.default_rng(seed_cfg)
    p0 = theta0 + 1e-4 * rng.standard_normal((nwalkers, ndim))

    with performance.blas_single_thread_ctx():
        performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(flat=True)

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    params = {"theta": mean, "cov": cov, "samples": samples}
    meta = {"nwalkers": nwalkers, "nsteps": nsteps}

    return UncReport(
        type="bayesian",
        params=params,
        curve_band={},
        meta=meta,
    )
