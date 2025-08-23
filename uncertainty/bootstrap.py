"""Bootstrap uncertainty estimation supporting several resampling schemes."""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class UncReport(TypedDict):
    type: str
    base_solver: str
    params: dict
    curve_band: dict
    meta: dict


def bootstrap(base_solver: str, resample_cfg: dict, residual_builder) -> UncReport:
    """Run a bootstrap procedure using ``base_solver``.

    ``resample_cfg`` describes the bootstrap flavour and number of resamples.
    The implementation is a placeholder to be fleshed out later.
    """
    raise NotImplementedError("Bootstrap uncertainty not yet implemented")
