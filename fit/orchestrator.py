"""High level fitting orchestration with solver fallbacks."""
from __future__ import annotations

from typing import Sequence, List

import numpy as np

from . import solve, iterate
from .result import FitResult

FALLBACKS = ["modern_vp", "modern_trf", "classic", "lmfit_vp"]


def run_fit_with_fallbacks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> FitResult:
    """Run the requested solver with simple fallbacks.

    Parameters
    ----------
    x, y : ``np.ndarray``
        Data to fit.
    peaks : sequence
        Initial peak guesses.
    mode : {"add", "subtract"}
        Baseline handling mode.
    baseline : ``np.ndarray`` or ``None``
        Baseline array matching ``x`` or ``None``.
    options : dict
        Dictionary of solver options.  Must contain a ``solver`` key selecting
        the primary solver.  Any remaining options are passed through to the
        backend ``solve`` call.
    """

    solver = options.get("solver", FALLBACKS[0])
    order: List[str] = []
    if solver in FALLBACKS:
        idx = FALLBACKS.index(solver)
        order = FALLBACKS[idx:] + FALLBACKS[:idx]
    else:  # pragma: no cover - defensive
        order = FALLBACKS

    last: FitResult | None = None
    msgs: List[str] = []
    for name in order:
        opts = dict(options)
        opts["solver"] = name
        result = solve(x, y, peaks, mode, baseline, opts)
        msgs.append(f"{name}:{'ok' if result.success else 'fail'}")
        if result.success:
            result.message = (result.message + " | " + " -> ".join(msgs)).strip()
            return result
        last = result
    # If we get here no solver succeeded; return the last result so the caller
    # still receives diagnostics.
    assert last is not None  # for type checkers
    last.message = (last.message + " | " + " -> ".join(msgs)).strip()
    return last


def step_once(state: dict) -> dict:
    """Run a single solver iteration without fallbacks."""

    return iterate(state)
