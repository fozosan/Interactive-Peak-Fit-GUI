import numpy as np


def noise_weights(y, mode, eps=1e-12, w_min=1e-8, w_max=1e8):
    """Return per-point noise weights according to ``mode``.

    Parameters
    ----------
    y : array_like
        Observed data after applying the chosen baseline.
    mode : {"none", "poisson", "inv_y"}
        Weighting strategy.
    eps : float, optional
        Small positive floor guarding against division by zero.
    w_min, w_max : float, optional
        Lower/upper clipping bounds for the weights.  Extreme values can lead
        to numerical instabilities inside the solvers.  The defaults are wide
        enough to be effectively no-ops for typical data ranges while still
        protecting against divisions by zero or overflow.
    """

    arr = np.asarray(y, dtype=float)
    if mode == "none":
        w = np.ones_like(arr)
    else:
        abs_y = np.maximum(np.abs(arr), eps)
        if mode == "poisson":
            w = 1.0 / np.sqrt(abs_y)
        elif mode == "inv_y":
            w = 1.0 / abs_y
        else:  # pragma: no cover - unknown mode
            raise ValueError(f"unknown mode '{mode}'")
    w = np.clip(w, w_min, w_max)
    w[~np.isfinite(w)] = 1.0
    return w


def robust_weights(r, loss, f_scale, eps=1e-12):
    """Return IRLS weights for residuals under ``loss``.

    Parameters
    ----------
    r : array_like
        Residual vector.
    loss : {"linear", "soft_l1", "huber", "cauchy"}
        Robust loss name following ``scipy.optimize.least_squares``.
    f_scale : float
        Scale parameter for the robust loss.
    eps : float, optional
        Small positive value guarding against division by zero.
    """
    r = np.asarray(r, dtype=float)
    fs = float(max(f_scale, eps))
    z = r / fs
    if loss == "linear":
        w = np.ones_like(z)
    elif loss == "soft_l1":
        w = (1.0 + z ** 2) ** -0.25
    elif loss == "huber":
        w = np.ones_like(z)
        mask = np.abs(z) > 1.0
        w[mask] = np.sqrt(1.0 / np.maximum(np.abs(z[mask]), eps))
    elif loss == "cauchy":
        w = (1.0 + z ** 2) ** -0.5
    else:  # pragma: no cover - unknown loss
        raise ValueError(f"unknown loss '{loss}'")
    w[~np.isfinite(w)] = 1.0
    return w


def combine_weights(w_noise: np.ndarray, w_rob: np.ndarray) -> np.ndarray:
    """Combine noise and robust weights into a single vector.

    Both input arrays are broadcast together element-wise and the product is
    rescaled so that the maximum weight equals ``1``.  Any non-finite values are
    replaced with ``1`` which effectively disables weighting for those points.
    """

    wn = np.asarray(w_noise, dtype=float)
    wr = np.asarray(w_rob, dtype=float)
    w = wn * wr
    w[~np.isfinite(w)] = 1.0
    if w.size:
        m = np.max(w)
        if m > 0:
            w = w / m
        else:  # all zeros
            w = np.ones_like(w)
    else:
        w = np.ones_like(w)
    return w

