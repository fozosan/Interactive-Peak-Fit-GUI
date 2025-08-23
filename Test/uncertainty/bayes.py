"""Bayesian uncertainty estimation via MCMC sampling."""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class UncReport(TypedDict):
    type: str
    params: dict
    curve_band: dict
    meta: dict


def bayesian(priors: dict, like: str, init_from_solver: dict,
             sampler_cfg: dict, constraints: dict | None) -> UncReport:
    """Sample parameter posteriors using ``emcee``.

    The actual sampling routine is not implemented in this stub.
    """
    raise NotImplementedError("Bayesian uncertainty not yet implemented")