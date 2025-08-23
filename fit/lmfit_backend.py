"""LMFIT backend providing parameter constraints and alternative algorithms."""
from __future__ import annotations


from typing import Optional, Sequence, TypedDict

import numpy as np

from core.peaks import Peak
from core.models import pv_sum



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
    peaks: list,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    """Fit peaks using the optional `lmfit` dependency.

    If `lmfit` is not installed, ``ok`` will be ``False`` and ``message`` will
    explain the missing dependency.
    """

    try:
        import lmfit
    except Exception as exc:  # pragma: no cover - lmfit missing
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

    params = lmfit.Parameters()
    for i, p in enumerate(peaks):
        params.add(f"c{i}", value=p.center, vary=not p.lock_center)
        params.add(f"h{i}", value=p.height)
        params.add(f"w{i}", value=p.fwhm, min=0, vary=not p.lock_width)
        params.add(f"e{i}", value=p.eta, min=0, max=1)

    def residual(pars: lmfit.Parameters) -> np.ndarray:
        pk: list[Peak] = []
        for i in range(len(peaks)):
            pk.append(
                Peak(
                    pars[f"c{i}"].value,
                    pars[f"h{i}"].value,
                    pars[f"w{i}"].value,
                    pars[f"e{i}"].value,
                )
            )
        model = pv_sum(x, pk)
        base = baseline if baseline is not None else 0.0
        if mode == "add":
            r = model + base - y
        elif mode == "subtract":
            r = model - (y - base)
        else:  # pragma: no cover - unknown mode
            raise ValueError("unknown mode")
        return r

    algo = options.get("algo", "least_squares")
    maxfev = int(options.get("maxfev", 20000))

    res = lmfit.minimize(residual, params, method=algo, max_nfev=maxfev)

    theta = []
    for i in range(len(peaks)):
        theta.extend(
            [
                res.params[f"c{i}"].value,
                res.params[f"h{i}"].value,
                res.params[f"w{i}"].value,
                res.params[f"e{i}"].value,
            ]
        )
    theta = np.asarray(theta, dtype=float)
    r = residual(res.params)
    cost = 0.5 * float(r @ r)

    return SolveResult(
        ok=res.success,
        theta=theta,
        message=res.message,
        cost=cost,
        jac=None,
        cov=res.covar,
        meta={"nfev": res.nfev},
    )

