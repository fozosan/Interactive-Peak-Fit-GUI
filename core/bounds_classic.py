import numpy as np


def make_bounds_classic(x_fit, y_target, peaks, fit_window=None, fwhm_min=None):
    x_lo, x_hi = float(np.min(x_fit)), float(np.max(x_fit))
    if fit_window:
        x_lo, x_hi = float(min(fit_window)), float(max(fit_window))

    dx = np.median(np.diff(np.sort(x_fit)))
    fwhm_lo = max(fwhm_min or 0.0, 2.0*dx)
    fwhm_hi = max(1e-6, x_hi - x_lo)

    p95 = float(np.percentile(y_target, 95))
    h_lo, h_hi = 0.0, max(1e-12, 2.0*p95, float(np.max(y_target)))

    # pack respecting locks (height always varied; center/width only if unlocked)
    p0, lo, hi, struct = [], [], [], []
    for pk in peaks:
        s = {}
        h0 = min(pk.height, h_hi) * 0.999
        s["ih"] = len(p0); p0.append(max(h0, 1e-9)); lo.append(h_lo); hi.append(h_hi)
        if pk.lock_center:
            s["ic"] = None; s["c_fixed"] = pk.center
        else:
            s["ic"] = len(p0); p0.append(pk.center); lo.append(x_lo); hi.append(x_hi)
        if pk.lock_width:
            s["iw"] = None; s["w_fixed"] = pk.fwhm
        else:
            s["iw"] = len(p0); p0.append(max(pk.fwhm, fwhm_lo)); lo.append(fwhm_lo); hi.append(fwhm_hi); s["w_fixed"] = None
        s["eta"] = float(np.clip(pk.eta, 0.0, 1.0))
        struct.append(s)

    p0 = np.asarray(p0, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    # clip p0 into bounds to avoid instant infeasible starts
    p0 = np.minimum(np.maximum(p0, lo), hi)
    return p0, (lo, hi), struct
