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

    # ``theta_list`` collects the starting parameter vector.  We ensure all
    # starting values already respect the bounds so the solvers do not start
    # outside the feasible region which can lead to immediate failures or very
    # poor convergence (observed when loading templates or using auto-seeded
    # peaks).

    theta_list: list[float] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    for pk in peaks:
        # Clamp starting values to lie inside the declared bounds.  Heights and
        # widths must be positive; eta is restricted to [0, 1]; centres can be
        # optionally clamped to the data window.  Doing this here keeps all
        # solvers consistent (Classic/Modern/LMFIT) as they all rely on this
        # helper to form ``theta0``.
        c = float(pk.center)
        if clamp_center:
            c = float(np.clip(c, x_min, x_max))
        h = float(max(pk.height, 1e-12))
        w = float(max(pk.fwhm, min_fwhm))
        e = float(np.clip(pk.eta, 0.0, 1.0))

        theta_list.extend([c, h, w, e])

        # bounds for center
        if pk.lock_center:
            lb_list.append(c)
            ub_list.append(c)
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
            lb_list.append(w)
            ub_list.append(w)
        else:
            lb_list.append(min_fwhm)
            ub_list.append(np.inf)

        # bounds for eta
        lb_list.append(0.0)
        ub_list.append(1.0)

    theta = np.asarray(theta_list, dtype=float)
    lb = np.asarray(lb_list, dtype=float)
    ub = np.asarray(ub_list, dtype=float)

    # ``scipy.optimize.least_squares`` requires ``lb < ub`` for all parameters
    # (strict inequality).  Locked parameters yield equal bounds which previously
    # triggered the "Each lower bound must be strictly less than each upper
    # bound" error when fitting from templates.  Add a tiny epsilon to the upper
    # bound to keep the parameter effectively fixed while satisfying the solver
    # requirement.
    mask = lb >= ub
    if np.any(mask):
        ub[mask] = lb[mask] + 1e-12

    return theta, (lb, ub)
