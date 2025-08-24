import numpy as np


def make_bounds(peaks, x, fit_mask=None, *, bound_centers_to_window=True,
                max_fwhm_factor=0.5, max_height_factor=2.0, y_target=None, **opts):
    x = np.asarray(x, dtype=float)
    if fit_mask is not None and np.any(fit_mask):
        x_fit = x[fit_mask]
    else:
        x_fit = x
    x_fit = np.asarray(x_fit, dtype=float)
    xmin = float(np.min(x_fit)) if x_fit.size else 0.0
    xmax = float(np.max(x_fit)) if x_fit.size else 0.0
    if x_fit.size > 1:
        dx = float(np.median(np.diff(np.sort(x_fit))))
    else:
        dx = 1.0
    fwhm_min = max(2.0 * dx, 1e-6)
    span = xmax - xmin
    if span > 0:
        fwhm_max = max_fwhm_factor * span
    else:
        fwhm_max = np.inf

    if 'max_height' in opts and opts['max_height'] is not None:
        h_cap = float(opts['max_height'])
    elif y_target is not None and np.asarray(y_target).size:
        y_arr = np.asarray(y_target, dtype=float)
        p95 = float(np.nanpercentile(np.abs(y_arr), 95))
        h_cap = max_height_factor * max(p95, 1e-9)
    else:
        h_cap = np.inf

    lo = []
    hi = []
    for pk in peaks:
        # height
        lo.append(0.0)
        hi.append(h_cap)
        # center
        if getattr(pk, 'lock_center', False):
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            if bound_centers_to_window:
                lo.append(xmin)
                hi.append(xmax)
            else:
                pad = 0.1 * span
                lo.append(xmin - pad)
                hi.append(xmax + pad)
        # width
        if getattr(pk, 'lock_width', False):
            lo.append(1e-6)
            hi.append(np.inf)
        else:
            lo.append(fwhm_min)
            hi.append(fwhm_max if np.isfinite(fwhm_max) else np.inf)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo[~np.isfinite(lo)] = -1e12
    hi[~np.isfinite(hi)] = 1e12
    hi = np.maximum(hi, lo + 1e-12)
    info = dict(fwhm_min=fwhm_min, fwhm_max=fwhm_max, h_cap=h_cap, xmin=xmin, xmax=xmax)
    return (lo, hi), info


def make_bounds_classic(x, y, peaks, fit_mask=None, cfg=None):
    """Return bounds and clipped start vector for the Classic solver.

    Parameters
    ----------
    x, y : array-like
        Data coordinates and target intensities (baseline already applied as
        required).  ``y`` may be ``None`` when only geometric bounds are
        desired.
    peaks : sequence
        Initial peak guesses.
    fit_mask : array-like or ``None``
        Boolean mask selecting the active fitting window within ``x``.
    cfg : mapping
        Classic configuration with keys ``bound_centers_to_window``,
        ``fwhm_min_dx_factor``, ``fwhm_max_span_factor`` and
        ``max_height_factor``.

    Returns
    -------
    (lo, hi) : tuple of ``np.ndarray``
        Lower and upper bounds for the flattened parameter vector ordered as
        ``[c, h, w, eta, ...]``.
    theta0 : ``np.ndarray``
        Starting parameter vector clipped into ``[lo, hi]``.
    """

    cfg = cfg or {}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) if y is not None else np.array([], dtype=float)

    if fit_mask is not None and np.any(fit_mask):
        x_fit = x[fit_mask]
        y_fit = y[fit_mask] if y.size else y
    else:
        x_fit = x
        y_fit = y

    xmin = float(np.min(x_fit)) if x_fit.size else 0.0
    xmax = float(np.max(x_fit)) if x_fit.size else 0.0
    span = xmax - xmin
    dx = float(np.median(np.diff(np.sort(x_fit)))) if x_fit.size > 1 else 1.0

    wmin = max(float(cfg.get("fwhm_min_dx_factor", 2.0)) * dx, 1e-6)
    wmax = (
        float(cfg.get("fwhm_max_span_factor", 0.5)) * span if span > 0 else np.inf
    )

    if y_fit.size:
        p95 = float(np.nanpercentile(y_fit, 95))
        p05 = float(np.nanpercentile(y_fit, 5))
        hmax = float(cfg.get("max_height_factor", 3.0)) * max(p95 - p05, 1e-9)
    else:
        hmax = np.inf

    bound_centers = bool(cfg.get("bound_centers_to_window", True))

    lo = []
    hi = []
    theta0 = []
    for pk in peaks:
        c = float(pk.center)
        h = float(pk.height)
        w = float(pk.fwhm)
        e = float(pk.eta)

        if getattr(pk, "lock_center", False):
            lo_c = hi_c = c
        else:
            if bound_centers:
                lo_c, hi_c = xmin, xmax
            else:
                pad = span
                lo_c, hi_c = xmin - pad, xmax + pad
            c = float(np.clip(c, lo_c, hi_c))

        if getattr(pk, "lock_height", False):
            lo_h = hi_h = h
        else:
            lo_h, hi_h = 0.0, hmax
            h = float(np.clip(h, lo_h, hi_h))

        if getattr(pk, "lock_width", False):
            lo_w = hi_w = w
        else:
            lo_w, hi_w = wmin, wmax
            w = float(np.clip(w, lo_w, hi_w))

        lo_e, hi_e = 0.0, 1.0
        e = float(np.clip(e, lo_e, hi_e))

        theta0.extend([c, h, w, e])
        lo.extend([lo_c, lo_h, lo_w, lo_e])
        hi.extend([hi_c, hi_h, hi_w, hi_e])

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)

    lo[~np.isfinite(lo)] = -1e12
    hi[~np.isfinite(hi)] = 1e12
    hi = np.maximum(hi, lo + 1e-12)
    theta0[~np.isfinite(theta0)] = 0.0
    theta0 = np.minimum(np.maximum(theta0, lo), hi)

    return (lo, hi), theta0
