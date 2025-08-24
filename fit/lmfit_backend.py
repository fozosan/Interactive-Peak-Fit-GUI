"""LMFIT backend using analytic residuals and Jacobian."""

from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np

from core.peaks import Peak
from core.residuals import build_residual_jac
from .bounds import pack_theta_bounds


class SolveResult(TypedDict):
    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict


def _theta_from_peaks(peaks: Sequence[Peak]) -> np.ndarray:
    arr: list[float] = []
    for p in peaks:
        arr.extend([p.center, p.height, p.fwhm, p.eta])
    return np.asarray(arr, dtype=float)


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    try:
        import lmfit
    except Exception as exc:  # pragma: no cover - dependency missing
        return SolveResult(
            ok=False,
            theta=_theta_from_peaks(peaks),
            message=str(exc),
            cost=float("nan"),
            jac=None,
            cov=None,
            meta={},
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    weight_mode = options.get("weights", "none")
    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(np.abs(y), 1e-12, None)

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    min_fwhm = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)
    x_min = float(x.min())
    x_max = float(x.max())
    clamp_center = bool(options.get("centers_in_window", False))

    params = lmfit.Parameters()
    for i, p in enumerate(peaks):
        c = float(np.clip(p.center, x_min, x_max)) if clamp_center else float(p.center)
        w = float(max(p.fwhm, min_fwhm))
        params.add(f"h{i}", value=max(p.height, 1e-12), min=0)
        params.add(
            f"c{i}", value=c,
            min=x_min if clamp_center else None,
            max=x_max if clamp_center else None,
            vary=not p.lock_center,
        )
        params.add(
            f"w{i}", value=w, min=min_fwhm, vary=not p.lock_width
        )

    resid_jac = build_residual_jac(x, y, peaks, mode, baseline, weights)

    def residual(pars: lmfit.Parameters) -> np.ndarray:
        theta = []
        for i, p in enumerate(peaks):
            theta.append(pars[f"h{i}"].value)
            if not p.lock_center:
                theta.append(pars[f"c{i}"].value)
            if not p.lock_width:
                theta.append(pars[f"w{i}"].value)
        r, _ = resid_jac(np.asarray(theta, dtype=float))
        return r

    def jac(pars: lmfit.Parameters) -> np.ndarray:
        theta = []
        for i, p in enumerate(peaks):
            theta.append(pars[f"h{i}"].value)
            if not p.lock_center:
                theta.append(pars[f"c{i}"].value)
            if not p.lock_width:
                theta.append(pars[f"w{i}"].value)
        _, J = resid_jac(np.asarray(theta, dtype=float))
        return J

    maxfev = int(options.get("maxfev", 20000))
    minimizer = lmfit.Minimizer(residual, params, jac=jac, nan_policy="omit")
    result = minimizer.minimize(method="least_squares", max_nfev=maxfev)

    theta_full = theta0_full.copy()
    idx = 0
    for i, p in enumerate(peaks):
        theta_full[4 * i + 1] = result.params[f"h{i}"].value
        if not p.lock_center:
            theta_full[4 * i + 0] = result.params[f"c{i}"].value
        if not p.lock_width:
            theta_full[4 * i + 2] = result.params[f"w{i}"].value
        # eta stays as initial

    r = residual(result.params)
    cost = 0.5 * float(r @ r)
    J = jac(result.params)

    return SolveResult(
        ok=result.success,
        theta=theta_full,
        message=result.message,
        cost=cost,
        jac=J,
        cov=result.covar,
        meta={"nfev": result.nfev},
    )

