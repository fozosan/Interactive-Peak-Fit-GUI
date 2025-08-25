import numpy as np


def noise_weights(mode, y):
    if mode == "none":
        return None
    y = np.asarray(y, float)
    if mode == "poisson":
        w = 1.0/np.sqrt(np.clip(np.abs(y), 1e-12, None))
    elif mode == "inv_y":
        w = 1.0/np.clip(np.abs(y), 1e-12, None)
    else:
        return None
    w[~np.isfinite(w)] = 0.0
    return w


def robust_weights(loss, r, f_scale=1.0):
    # Return None for linear, otherwise IRLS weights (Huber/soft_l1/cauchy)
    if loss in (None, "", "linear"):
        return None
    r = np.asarray(r, float) / max(1e-12, float(f_scale))
    a = np.abs(r)
    if loss == "soft_l1":
        w = 1.0/np.sqrt(1.0 + a*a)
    elif loss == "huber":
        k = 1.0
        w = np.where(a <= k, 1.0, k/a)
    elif loss == "cauchy":
        w = 1.0/(1.0 + a*a)
    else:
        return None
    w[~np.isfinite(w)] = 0.0
    return w


def combine_weights(w_noise, w_robust):
    if w_noise is None and w_robust is None:
        return None
    if w_noise is None:
        return w_robust
    if w_robust is None:
        return w_noise
    w = w_noise * w_robust
    w[~np.isfinite(w)] = 0.0
    return w
