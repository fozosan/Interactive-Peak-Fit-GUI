from __future__ import annotations

"""Utilities for packing peak parameters and bounds."""

from typing import Sequence, Tuple

import numpy as np

from core.peaks import Peak


def pack_theta_bounds(
    peaks: Sequence[Peak],
    x: np.ndarray,
    options: dict | None = None,
) -> Tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return flattened parameters and corresponding bounds.

    Parameters
    ----------
    peaks:
        Iterable of :class:`~core.peaks.Peak` instances describing the
        current model.
    x:
        1D array of x-values used for bounding peak centers when
        ``options['centers_in_window']`` is true.
    options:
        Mapping of solver options. Recognised keys are ``centers_in_window``
        and ``min_fwhm``.

    Returns
    -------
    theta, bounds
        ``theta`` is the flattened parameter vector ``[c1, h1, w1, e1, c2, ...]``.
        ``bounds`` is a ``(lb, ub)`` tuple matching ``theta``.
    """

    options = options or {}
    x = np.asarray(x, dtype=float)
    theta = []
    lb = []
    ub = []

    x_min = float(x.min()) if x.size else -np.inf
    x_max = float(x.max()) if x.size else np.inf
    min_fwhm = float(options.get("min_fwhm", 1e-6))
    clamp_centers = bool(options.get("centers_in_window", False))

    for pk in peaks:
        theta.extend([pk.center, pk.height, pk.fwhm, pk.eta])

        # center bounds
        if pk.lock_center:
            c_lb = c_ub = pk.center
        elif clamp_centers:
            c_lb, c_ub = x_min, x_max
        else:
            c_lb, c_ub = -np.inf, np.inf
        lb.append(c_lb)
        ub.append(c_ub)

        # height bounds (non-negative)
        lb.append(0.0)
        ub.append(np.inf)

        # fwhm bounds
        if pk.lock_width:
            lb.append(pk.fwhm)
            ub.append(pk.fwhm)
        else:
            lb.append(min_fwhm)
            ub.append(np.inf)

        # eta bounds (0-1)
        lb.append(0.0)
        ub.append(1.0)

    theta_arr = np.asarray(theta, dtype=float)
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    return theta_arr, (lb_arr, ub_arr)


__all__ = ["pack_theta_bounds"]
