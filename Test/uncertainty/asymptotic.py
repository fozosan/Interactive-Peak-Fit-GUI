"""Asymptotic uncertainty estimation based on the Jacobian."""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class UncReport(TypedDict):
    type: str
    params: dict
    curve_band: dict
    meta: dict


def asymptotic(report_from_solver: dict, residual_builder) -> UncReport:
    """Return a curvature-based uncertainty report.

    ``report_from_solver`` is expected to match the ``SolveResult`` structure.
    The implementation is intentionally left as a placeholder.
    """
    raise NotImplementedError("Asymptotic uncertainty not yet implemented")