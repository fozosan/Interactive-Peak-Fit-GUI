import numpy as np


def noise_weights(y, mode, eps=1e-12):
    """Return per-point noise weights according to ``mode``.

    Parameters
    ----------
    y : array_like
        Observed data after applying the chosen baseline.
    mode : {"none", "poisson", "inv_y"}
        Weighting strategy.
    eps : float, optional
        Small positive floor guarding against division by zero.
    """
    arr = np.asarray(y, dtype=float)
    if mode == "none":
        return np.ones_like(arr)
    abs_y = np.maximum(np.abs(arr), eps)
    if mode == "poisson":
        return 1.0 / np.sqrt(abs_y)
    if mode == "inv_y":
        return 1.0 / abs_y
    raise ValueError(f"unknown mode '{mode}'")


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
    return w
