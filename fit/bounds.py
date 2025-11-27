"""Parameter packing and bounds computation for peak fitting."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from core.bounds import make_bounds
import logging
log = logging.getLogger(__name__)


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
    span = float(x.max() - x.min()) if x.size else 0.0
    opts = dict(options)
    bound_centers = bool(
        opts.get("centers_in_window", opts.get("bound_centers_to_window", False))
    )
    max_fwhm_factor = float(opts.get("max_fwhm_factor", 0.5))
    if "max_fwhm" in opts and span > 0:
        max_fwhm_factor = float(opts["max_fwhm"]) / span
    max_height_factor = float(opts.get("max_height_factor", np.inf))
    y_target = opts.get("y_target")
    extra = {"max_height": opts.get("max_height")}
    (lo_hw, hi_hw), _info = make_bounds(
        peaks,
        x,
        None,
        bound_centers_to_window=bound_centers,
        max_fwhm_factor=max_fwhm_factor,
        max_height_factor=max_height_factor,
        y_target=y_target,
        **extra,
    )

    # ``theta_list`` collects the starting parameter vector.  We ensure all
    # starting values already respect the bounds so the solvers do not start
    # outside the feasible region which can lead to immediate failures or very
    # poor convergence (observed when loading templates or using auto-seeded
    # peaks).

    theta_list: list[float] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    caps = options.get("width_caps", None)
    for i, pk in enumerate(peaks):
        h_lo, c_lo, w_lo = lo_hw[3 * i : 3 * i + 3]
        h_hi, c_hi, w_hi = hi_hw[3 * i : 3 * i + 3]

        c = float(np.clip(pk.center, c_lo, c_hi))
        h = float(np.clip(max(pk.height, 1e-12), h_lo, h_hi))
        w = float(np.clip(max(pk.fwhm, w_lo), w_lo, w_hi))
        e = float(np.clip(pk.eta, 0.0, 1.0))

        theta_list.extend([c, h, w, e])

        # bounds for center
        if pk.lock_center:
            lb_list.append(c)
            ub_list.append(c)
        else:
            lb_list.append(c_lo)
            ub_list.append(c_hi)

        # bounds for height
        lb_list.append(h_lo)
        ub_list.append(h_hi)

        # bounds for width
        if pk.lock_width:
            lb_list.append(w)
            ub_list.append(w)
        else:
            lb_list.append(w_lo)
            # Optional per-peak width cap
            if isinstance(caps, (list, tuple)) and i < len(caps):
                cap = caps[i]
                if cap is not None:
                    try:
                        capf = float(cap)
                        if np.isfinite(capf) and capf > 0:
                            old = w_hi
                            w_hi = min(w_hi, capf)
                            if w_lo >= w_hi:
                                w_lo = max(1e-9, min(w_lo, 0.999999 * w_hi))
                            try:
                                log.debug(f"[bounds] cap peak {i+1}: w_hi {old:.6g} → {w_hi:.6g}")
                            except Exception:
                                pass
                    except Exception:
                        pass
            ub_list.append(w_hi)

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
