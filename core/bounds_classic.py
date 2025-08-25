import numpy as np


def make_bounds_classic(
    x_fit,
    y_fit,
    peaks,
    *,
    centers_in_window: bool,
    fwhm_min_factor: float = 2.0,
    fwhm_max_factor: float = 0.5,
    height_max_factor: float = 1.0,
    margin_frac: float = 0.0,
):
    """Return simple bounds for Classic solver.

    Parameters are ordered ``height``, optional ``center`` (if unlocked), and
    optional ``width`` (if unlocked) for each peak. Only free parameters are
    included. ``wmin_eval`` is returned to clamp widths inside residual
    evaluation.
    """
    x = np.asarray(x_fit, float)
    y = np.asarray(y_fit, float)
    if x.size:
        x_lo = float(np.min(x))
        x_hi = float(np.max(x))
        span = x_hi - x_lo
        dx_med = np.median(np.diff(np.sort(x))) if x.size > 1 else 1.0
    else:  # pragma: no cover - degenerate input
        x_lo = x_hi = 0.0
        span = 1.0
        dx_med = 1.0
    wmin = max(fwhm_min_factor * dx_med, 1e-9)
    wmax = max(span * fwhm_max_factor, wmin)
    margin = span * margin_frac
    if centers_in_window:
        c_lo, c_hi = x_lo, x_hi
    else:
        c_lo, c_hi = x_lo - margin, x_hi + margin
    y_abs = np.abs(y[np.isfinite(y)])
    if y_abs.size:
        p95 = float(np.percentile(y_abs, 95))
    else:  # pragma: no cover - degenerate input
        p95 = 1.0
    hmax = height_max_factor * p95

    lo = []
    hi = []
    for pk in peaks:
        lo.append(0.0)
        hi.append(hmax)
        if not pk.lock_center:
            lo.append(c_lo)
            hi.append(c_hi)
        if not pk.lock_width:
            lo.append(wmin)
            hi.append(wmax)
    return np.asarray(lo, float), np.asarray(hi, float), float(wmin)
