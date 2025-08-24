"""Unified solver interface for Peakfit 3.x."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Callable, Sequence

import numpy as np

from core import models
from .result import FitResult
from . import classic, modern, modern_vp, lmfit_backend, step_engine, bounds

# Mapping of public solver names to backend implementations.  ``modern`` is the
# Trust Region Reflective solver; ``modern_vp`` uses variable projection.
_SOLVERS: Dict[str, Callable] = {
    "classic": classic.solve,
    "modern_trf": modern.solve,
    "modern_vp": modern_vp.solve,
    "lmfit_vp": lmfit_backend.solve,
}

_ITERATORS: Dict[str, Callable] = {
    "classic": classic.iterate,
    "modern_trf": modern.iterate,
    "modern_vp": modern_vp.iterate,
    "lmfit_vp": lmfit_backend.iterate,
}

__all__ = [
    "classic",
    "modern",
    "modern_vp",
    "lmfit_backend",
    "step_engine",
    "bounds",
    "solve",
    "iterate",
    "FitResult",
]


def _peaks_from_theta(peaks: Sequence, theta: np.ndarray) -> list:
    """Return deep-copied peaks updated from a flattened ``theta`` vector."""

    out = []
    for i, pk in enumerate(peaks):
        c, h, w, e = theta[4 * i : 4 * (i + 1)]
        # ``replace`` works for both :mod:`core.peaks` and the GUI's dataclass.
        out.append(
            replace(
                pk,
                center=float(c),
                height=float(h),
                fwhm=float(abs(w)),
                eta=float(e),
            )
        )
    return out


def _adapt_result(
    raw: dict,
    solver_name: str,
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence,
    mode: str,
    baseline: np.ndarray | None,
) -> FitResult:
    """Convert legacy ``SolveResult`` dictionaries into :class:`FitResult`."""

    theta = np.asarray(raw.get("theta", []), dtype=float)
    peaks_out = _peaks_from_theta(peaks, theta)
    base = baseline if baseline is not None else 0.0
    model = models.pv_sum(x, peaks_out)
    if mode == "add":
        resid = model + base - y
    else:
        resid = model - (y - base)
    rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    meta = raw.get("meta", {}) or {}
    nfev = int(meta.get("nfev", 0))
    n_iter = int(meta.get("n_iter", meta.get("njev", nfev)))
    return FitResult(
        success=bool(raw.get("ok", False)),
        solver=solver_name,
        theta=theta,
        peaks_out=peaks_out,
        cost=float(raw.get("cost", np.nan)),
        rmse=rmse,
        nfev=nfev,
        n_iter=n_iter,
        message=str(raw.get("message", "")),
        diagnostics=meta,
    )


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> FitResult:
    """Dispatch to the selected backend and return a :class:`FitResult`.

    The ``options`` mapping must contain a ``solver`` key identifying the
    backend (``classic`` | ``modern_vp`` | ``modern_trf`` | ``lmfit_vp``).  Any
    additional items are forwarded to the backend's native ``solve`` function.
    """

    solver_name = options.get("solver", "modern_vp")
    func = _SOLVERS.get(solver_name)
    if func is None:  # pragma: no cover - defensive programming
        raise ValueError(f"unknown solver '{solver_name}'")
    raw = func(x, y, peaks, mode, baseline, options)
    return _adapt_result(raw, solver_name, x, y, peaks, mode, baseline)


def iterate(state: dict) -> dict:
    """Dispatch to the backend iterate function."""

    opts = state.get("options", {})
    solver_name = opts.get("solver", "modern_vp")
    func = _ITERATORS.get(solver_name)
    if func is None:  # pragma: no cover - defensive programming
        raise ValueError(f"unknown solver '{solver_name}'")
    return func(state)
