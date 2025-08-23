"""Bootstrap uncertainty estimation supporting several resampling schemes."""
from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np

from core.peaks import Peak



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


def bootstrap(base_solver: str, resample_cfg: dict, residual_builder) -> UncReport:
    """Estimate parameter uncertainty via residual bootstrap.

    Parameters
    ----------
    base_solver:
        Which solver backend to use (``"classic"``, ``"modern"``, ``"lmfit"``).
    resample_cfg:
        Dictionary describing the problem. Expected keys are ``x``, ``y``,
        ``peaks`` (template peaks), ``mode``, ``baseline``, ``theta`` (final
        parameter vector), ``options`` (solver options), ``n`` (number of
        resamples) and optional ``seed``.
    residual_builder:
        Callable returning residuals for a parameter vector. It should match the
        problem described in ``resample_cfg``.
    """

    if base_solver == "classic":
        from fit.classic import solve as solver
    elif base_solver == "modern":
        from fit.modern import solve as solver
    elif base_solver == "lmfit":
        from fit.lmfit_backend import solve as solver
    else:  # pragma: no cover - unknown solver
        raise ValueError("unknown solver")

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

    rng = np.random.default_rng(resample_cfg.get("seed"))
    samples = []
    start_peaks = _peaks_from_theta(theta, peaks)
    for _ in range(n):
        resampled = rng.choice(r, size=r.size, replace=True)
        y_boot = fitted - resampled
        res = solver(x, y_boot, start_peaks, mode, baseline, options)
        samples.append(np.asarray(res["theta"], dtype=float))

    samples = np.vstack(samples) if samples else np.empty((0, theta.size))
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

