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

    strategy = str(
        sampler_cfg.get(
            "perf_parallel_strategy",
            init_from_solver.get("perf_parallel_strategy", "outer"),
        )
    )
    try:
        blas_threads = int(
            sampler_cfg.get(
                "perf_blas_threads",
                init_from_solver.get("perf_blas_threads", 0),
            )
            or 0
        )
    except Exception:
        blas_threads = 0
    limit = 1 if strategy == "outer" else (blas_threads if blas_threads > 0 else None)

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
    try:
        seed_int = int(seed_cfg) if seed_cfg is not None else None
    except Exception:
        seed_int = None
    seed_effective = (
        seed_int if seed_int is not None else (cfg_perf.seed_value if cfg_perf.seed_all else None)
    )
    rng = np.random.default_rng(seed_effective)
    p0 = theta0 + 1e-4 * rng.standard_normal((nwalkers, ndim))

    performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
    with performance.blas_limit_ctx(limit):
        try:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_prob,
                random_state=seed_effective if seed_effective is not None else None,
            )
        except TypeError:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            if seed_effective is not None:
                try:
                    sampler.random_state = np.random.RandomState(int(seed_effective))
                except Exception:
                    pass
        performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
        sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(flat=True)

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    params = {"theta": mean, "cov": cov, "samples": samples}
    meta = {
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "seed": (int(seed_effective) if seed_effective is not None else seed_cfg),
        "seed_all": bool(cfg_perf.seed_all),
        "parallel_strategy": strategy,
        "blas_threads": int(blas_threads),
        "blas_effective": (
            1 if strategy == "outer" else (int(blas_threads) if blas_threads > 0 else None)
        ),
        "numpy_backend": performance.which_backend(),
    }

    return UncReport(
        type="bayesian",
        params=params,
        curve_band={},
        meta=meta,
    )
