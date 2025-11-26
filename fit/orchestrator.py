"""High level fitting orchestration without implicit solver fallbacks."""
from __future__ import annotations

from typing import Sequence, List

import numpy as np

from . import solve, iterate
from .result import FitResult

FALLBACKS: list[str] = []  # disable fallbacks: only the requested solver should run


def run_fit_with_fallbacks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence,
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> FitResult:
    """Run the requested solver without attempting implicit fallbacks."""

    if "solver" not in options or not options["solver"]:
        raise KeyError("options must include a non-empty 'solver' entry")

    opts = dict(options)
    result = solve(x, y, peaks, mode, baseline, opts)
    return result


def step_once(state: dict) -> dict:
    """Run a single solver iteration without fallbacks."""

    return iterate(state)
