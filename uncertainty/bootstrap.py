"""Bootstrap uncertainty estimation supporting several resampling schemes."""
from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from core.peaks import Peak
from infra import performance


class UncReport(TypedDict):
    type: str
    base_solver: str
    params: dict
    curve_band: dict
    meta: dict


def _peaks_from_theta(theta: np.ndarray, template: Sequence[Peak]) -> list[Peak]:
    pk: list[Peak] = []
    for i, tpl in enumerate(template):
        c, h, fw, eta = theta[4 * i : 4 * (i + 1)]
        pk.append(Peak(c, h, fw, eta, tpl.lock_center, tpl.lock_width))
    return pk


def _bootstrap_worker(args):
    (
        base_solver,
        x,
        fitted,
        resid,
        start_peaks,
        mode,
        baseline,
        options,
        seed_base,
        idx,
    ) = args

    if performance._SEED is not None:
        np.random.seed(performance._SEED + idx)

    if base_solver == "classic":
        from fit.classic import solve as solver
    elif base_solver == "modern_vp":
        from fit.modern_vp import solve as solver
    elif base_solver == "modern_trf":
        from fit.modern import solve as solver
    elif base_solver == "lmfit_vp":
        from fit.lmfit_backend import solve as solver
    else:  # pragma: no cover - unknown solver
        raise ValueError("unknown solver")

    rng = np.random.default_rng(seed_base + idx if seed_base is not None else None)
    resampled = rng.choice(resid, size=resid.size, replace=True)
    y_boot = fitted - resampled
    pk_copy = [
        Peak(p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width)
        for p in start_peaks
    ]
    res = solver(x, y_boot, pk_copy, mode, baseline, options)
    return np.asarray(res["theta"], dtype=float)


def bootstrap(base_solver: str, resample_cfg: dict, residual_builder) -> UncReport:
    """Estimate parameter uncertainty via residual bootstrap.

    Parameters
    ----------
    base_solver:
        Which solver backend to use (``"classic"``, ``"modern_vp"``,
        ``"modern_trf"``, ``"lmfit_vp"``).
    resample_cfg:
        Dictionary describing the problem. Expected keys are ``x``, ``y``,
        ``peaks`` (template peaks), ``mode``, ``baseline``, ``theta`` (final
        parameter vector), ``options`` (solver options), ``n`` (number of
        resamples) and optional ``seed``.
    residual_builder:
        Callable returning residuals for a parameter vector. It should match the
        problem described in ``resample_cfg``.
    """

    x = np.asarray(resample_cfg["x"], dtype=float)
    y = np.asarray(resample_cfg["y"], dtype=float)
    peaks = list(resample_cfg["peaks"])  # template peaks
    mode = resample_cfg.get("mode", "add")
    baseline = (
        np.asarray(resample_cfg.get("baseline"), dtype=float)
        if resample_cfg.get("baseline") is not None
        else None
    )
    options = resample_cfg.get("options", {})
    n = int(resample_cfg.get("n", 100))
    theta = np.asarray(resample_cfg["theta"], dtype=float)

    resid_fn = residual_builder
    r = resid_fn(theta)
    fitted = y + r

    seed_base = resample_cfg.get("seed")
    start_peaks = _peaks_from_theta(theta, peaks)

    cfg_perf = performance.get_parallel_config()
    try:
        workers_req = int(resample_cfg.get("workers", 0) or 0)
    except Exception:
        workers_req = 0
    draw_workers = workers_req if workers_req > 0 else cfg_perf.unc_workers
    draw_workers = max(1, int(draw_workers))
    args_common = (base_solver, x, fitted, r, start_peaks, mode, baseline, options, seed_base)
    with performance.blas_single_thread_ctx():
        performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
        if draw_workers > 1:
            with ProcessPoolExecutor(max_workers=draw_workers) as ex:
                iter_args = (args_common + (i,) for i in range(n))
                samples_list = list(ex.map(_bootstrap_worker, iter_args))
        else:
            samples_list = [_bootstrap_worker(args_common + (i,)) for i in range(n)]

    samples = np.vstack(samples_list) if samples_list else np.empty((0, theta.size))
    mean_theta = samples.mean(axis=0) if samples.size else theta
    cov = (
        np.cov(samples, rowvar=False, ddof=1)
        if samples.shape[0] > 1
        else np.zeros((theta.size, theta.size))
    )

    params = {"theta": mean_theta, "cov": cov, "samples": samples}
    meta = {"n": n}

    return UncReport(
        type="bootstrap",
        base_solver=base_solver,
        params=params,
        curve_band={},
        meta=meta,
    )
