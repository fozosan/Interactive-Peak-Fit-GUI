"""Parameter packing and bounds computation for peak fitting."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class PeakLike:
    """Protocol-like base for objects with peak attributes.

    ``fit.bounds`` works with both :class:`core.peaks.Peak` and the lightweight
    ``ui.app.Peak`` dataclasses used by the GUI.  We only rely on attribute
    access, so a formal ``Protocol`` is unnecessary and avoids an optional
    typing dependency."""

    center: float
    height: float
    fwhm: float
    eta: float
    lock_center: bool
    lock_width: bool


def pack_theta_bounds(
    peaks: Sequence[PeakLike],
    x: Sequence[float],
    options: dict,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Flatten ``peaks`` into a parameter vector and matching bounds.

    Parameters
    ----------
    peaks:
        Sequence of peak-like objects.  Each must provide ``center``, ``height``,
        ``fwhm`` and ``eta`` attributes plus ``lock_center``/``lock_width`` flags.
    x:
        Sample positions corresponding to the data being fitted.  Only the
        minimum and maximum are used to optionally constrain peak centers.
    options:
        Mapping of solver options.  ``min_fwhm`` (default ``1e-6``) enforces a
        lower bound on widths.  ``centers_in_window`` constrains centers to the
        range of ``x`` when true.

    Returns
    -------
    theta : ``np.ndarray``
        Flattened parameter vector ``[c0, h0, w0, e0, c1, h1, ...]``.
    bounds : tuple of ``np.ndarray``
        Lower and upper bounds suitable for scipy ``least_squares`` style APIs.
    """

    x = np.asarray(x, dtype=float)
    x_min = float(x.min())
    x_max = float(x.max())
    min_fwhm = float(options.get("min_fwhm", 1e-6))
    clamp_center = bool(options.get("centers_in_window", False))

    theta_list: list[float] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    for pk in peaks:
        # parameter packing
        theta_list.extend([pk.center, pk.height, pk.fwhm, pk.eta])

        # bounds for center
        if pk.lock_center:
            lb_list.append(pk.center)
            ub_list.append(pk.center)
        elif clamp_center:
            lb_list.append(x_min)
            ub_list.append(x_max)
        else:
            lb_list.append(-np.inf)
            ub_list.append(np.inf)

        # bounds for height
        lb_list.append(0.0)
        ub_list.append(np.inf)

        # bounds for width
        if pk.lock_width:
            lb_list.append(pk.fwhm)
            ub_list.append(pk.fwhm)
        else:
            lb_list.append(min_fwhm)
            ub_list.append(np.inf)

        # bounds for eta
        lb_list.append(0.0)
        ub_list.append(1.0)

    theta = np.asarray(theta_list, dtype=float)
    lb = np.asarray(lb_list, dtype=float)
    ub = np.asarray(ub_list, dtype=float)
    return theta, (lb, ub)
