"""Parameter packing and bounds computation for peak fitting."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from core.bounds import make_bounds
from core.models import pseudo_voigt
import logging
log = logging.getLogger(__name__)


# make_bounds' stand-in for "no bound": the bound of a height the fit keeps
# below the cap (fit_api.run_fit_consistent), as on main.
NO_BOUND = 1e12


def visible_share(center: float, fwhm: float, eta: float, xmin: float, xmax: float) -> float:
    """The largest value a unit-height peak takes over [xmin, xmax]: 1 when its
    centre lies inside, else its value at the nearer end of the window.

    A peak centred outside the window reaches the data with its tail only.
    pack_theta_bounds holds the height of a peak whose share is below sqrt(eps):
    no data can see it (a near-Gaussian locked 200 cm-1 away shows 1e-43 of
    itself), and its column would only feed the solvers noise."""

    if xmax <= xmin or xmin <= center <= xmax:
        return 1.0
    edge = xmin if center < xmin else xmax
    return float(pseudo_voigt(np.array([edge], dtype=float), 1.0, center, max(fwhm, 1e-12), eta)[0])


def height_cap(y_target, options=None, y_data=None) -> float:
    """The largest height a fit may give a peak: ``options["max_height"]`` when
    given (a bootstrap refit takes its fit's), else the data's largest value
    above the lowest the baseline can lie, plus the peak signal max|y_target|.

    ``y_target`` is the window's data less the baseline the peaks are fitted
    over: the fixed one, or the computed one a joint baseline starts from. A
    peak adds its height to the baseline at its centre, so it can be no taller
    than the data there above the lowest the baseline can lie, plus what the
    fit leaves in the residual, for which the peak signal is the allowance.

    - A fixed baseline lies where it is: the bound is twice the peak signal.
    - A joint baseline (``y_data``, the window's data, given) is fitted with the
      peaks and can dip below the data, a broad peak and the baseline trading
      against each other; it cannot dip below zero counts (or below the data,
      for data that go below zero). The bound is max(y_data) - min(0,
      min(y_data)) plus the peak signal: on the 5 s MoS2 spectrum 595 + 91,
      where twice the signal, 181, cut fits whose broad peaks reach 258 and 535.

    One value for every height. A height the fit keeps below it is bound by
    nothing (fit_api.run_fit_consistent), and one that ends at it makes the fit
    fail, by name."""

    opts = options or {}
    given = opts.get("max_height")
    if given is not None and np.isfinite(float(given)):
        return float(given)
    y = np.asarray(y_target, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if not y.size:
        return float("inf")
    signal = max(float(np.max(np.abs(y))), 1e-12)
    if y_data is None:
        return 2.0 * signal
    d = np.asarray(y_data, dtype=float).ravel()
    d = d[np.isfinite(d)]
    above = float(np.max(d)) - min(0.0, float(np.min(d))) if d.size else signal
    return max(above, 0.0) + signal


def held_heights(bounds) -> np.ndarray:
    """The value each height's bounds hold it at, NaN for a free height.

    pack_theta_bounds holds the height of a peak the window cannot see at its
    lower bound, as [lo, lo + 1e-12] (heights' lower bound is 0, so the pair is
    exact). Such a height is no parameter, as a locked centre or width is none:
    the solvers leave it out of the values they vary."""

    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    return np.where(ub[1::4] - lb[1::4] <= 1e-12, lb[1::4], np.nan)


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
        range of ``x`` when true.  ``max_height`` is the heights' upper bound
        (:func:`height_cap`), which every caller sets from the window's data;
        ``capped`` (one flag per peak; default: every peak) says which heights
        take it, the others keeping make_bounds' stand-in for none.

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
    capped = options.get("capped")
    xmin, xmax = (float(x.min()), float(x.max())) if x.size else (0.0, 0.0)
    for i, pk in enumerate(peaks):
        h_lo, c_lo, w_lo = lo_hw[3 * i : 3 * i + 3]
        h_hi, c_hi, w_hi = hi_hw[3 * i : 3 * i + 3]

        c = float(np.clip(pk.center, c_lo, c_hi))
        w = float(np.clip(max(pk.fwhm, w_lo), w_lo, w_hi))
        e = float(np.clip(pk.eta, 0.0, 1.0))

        # bounds for width
        if pk.lock_width:
            w_lo = w_hi = w
        else:
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

        # bounds for centre: a locked one stays where it is, in the window or not
        if pk.lock_center:
            c_lo = c_hi = c

        # bounds for height: the data's cap (height_cap) for a height that takes
        # it; one that does not keeps the stand-in for none. A peak whose largest
        # value in the window (visible_share at the allowed centre nearest the
        # window and the widest allowed width; eta is never fitted) is below
        # sqrt(eps) of a unit peak is held at 0 in every pass, cap or none: a
        # finite-difference derivative of it is rounding noise (Classic drove
        # such heights to 4e7) and its NNLS column as small (Modern VP: 4.6e15).
        if h_hi < NO_BOUND:  # a bound from the data, not make_bounds' stand-in for none
            c_near = float(np.clip(0.5 * (xmin + xmax), c_lo, c_hi))
            if visible_share(c_near, w_hi, e, xmin, xmax) < np.sqrt(np.finfo(float).eps):
                h_hi = h_lo
            elif capped is not None and not capped[i]:
                h_hi = NO_BOUND
        h = float(np.clip(max(pk.height, 1e-12), h_lo, h_hi))

        theta_list.extend([c, h, w, e])
        lb_list.extend([c_lo, h_lo, w_lo, 0.0])
        ub_list.extend([c_hi, h_hi, w_hi, 1.0])

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
