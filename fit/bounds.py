from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from core.peaks import Peak


def pack_theta_bounds(
    peaks: Sequence[Peak], x: np.ndarray, options: dict
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return initial parameter vector and bounds honoring locks and limits."""

    x = np.asarray(x, dtype=float)
    x_min = float(x.min()) if x.size else -np.inf
    x_max = float(x.max()) if x.size else np.inf
    min_fwhm = float(options.get("min_fwhm", 1e-6))
    clamp_center = bool(options.get("centers_in_window", False))

    theta0 = []
    lb = []
    ub = []

    eps = 1e-12
    for p in peaks:
        # center
        c = float(p.center)
        lb_c = x_min if clamp_center else -np.inf
        ub_c = x_max if clamp_center else np.inf
        if p.lock_center:
            lb_c = c - eps
            ub_c = c + eps
        theta0.append(np.clip(c, lb_c, ub_c))
        lb.append(lb_c)
        ub.append(ub_c)

        # height
        h = max(float(p.height), 0.0)
        theta0.append(h)
        lb.append(0.0)
        ub.append(np.inf)

        # width
        w = max(float(p.fwhm), min_fwhm)
        lb_w = min_fwhm
        ub_w = np.inf
        if p.lock_width:
            lb_w = w - eps
            ub_w = w + eps
        theta0.append(np.clip(w, lb_w, ub_w))
        lb.append(lb_w)
        ub.append(ub_w)

        # eta
        e = float(np.clip(p.eta, 0.0, 1.0))
        theta0.append(e)
        lb.append(0.0)
        ub.append(1.0)

    theta0_arr = np.asarray(theta0, dtype=float)
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    return theta0_arr, (lb_arr, ub_arr)
