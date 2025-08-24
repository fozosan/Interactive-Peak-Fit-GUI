"""Utilities for packing peak parameters and bounds."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from core.peaks import Peak


def pack_theta_bounds(
    peaks: Sequence[Peak],
    x: np.ndarray,
    options: dict,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return flattened peak parameters and bound arrays.

    Parameters
    ----------
    peaks:
        Sequence of :class:`~core.peaks.Peak` objects to flatten.
    x:
        Abscissa values used to optionally clamp peak centers.
    options:
        Mapping that may contain ``min_fwhm`` and ``centers_in_window`` keys
        controlling bounds.

    The parameter vector is ordered as ``[center, height, fwhm, eta]`` for each
    peak. Bounds enforce non-negative heights, a minimum full-width at half
    maximum, and 0 ≤ η ≤ 1. If ``centers_in_window`` is truthy, peak centers are
    limited to ``[x.min(), x.max()]``. When ``lock_center`` or ``lock_width`` is
    set on a peak the respective parameter is effectively fixed using a tiny
    tolerance around the value to satisfy solvers that require strictly
    increasing bounds. Initial parameter guesses are clipped into the valid
    range so that they always satisfy the returned bounds.
    """

    x = np.asarray(x, dtype=float)
    theta: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    x_min = float(x.min())
    x_max = float(x.max())
    min_fwhm = float(options.get("min_fwhm", 1e-6))
    clamp_center = bool(options.get("centers_in_window", False))
    eps = np.finfo(float).eps

    for pk in peaks:
        # center
        c = float(pk.center)
        if clamp_center and not pk.lock_center:
            c = float(np.clip(c, x_min, x_max))
        theta.append(c)
        if pk.lock_center:
            lb.append(c - eps)
            ub.append(c + eps)
        else:
            if clamp_center:
                lb.append(x_min)
                ub.append(x_max)
            else:
                lb.append(-np.inf)
                ub.append(np.inf)

        # height
        h = float(max(pk.height, 0.0))
        theta.append(h)
        lb.append(0.0)
        ub.append(np.inf)

        # FWHM
        w = float(max(pk.fwhm, min_fwhm))
        theta.append(w)
        if pk.lock_width:
            lb.append(w - eps)
            ub.append(w + eps)
        else:
            lb.append(min_fwhm)
            ub.append(np.inf)

        # eta (shape factor)
        e = float(np.clip(pk.eta, 0.0, 1.0))
        theta.append(e)
        lb.append(0.0)
        ub.append(1.0)

    theta_arr = np.asarray(theta, dtype=float)
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    return theta_arr, (lb_arr, ub_arr)
