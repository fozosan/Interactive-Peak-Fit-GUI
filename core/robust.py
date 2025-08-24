import numpy as np


def irls_weights(resid, loss, f_scale, eps=1e-12):
    """Return IRLS weights for residuals under ``loss``.

    Parameters
    ----------
    resid : array_like
        Residual vector.
    loss : {"linear", "soft_l1", "huber", "cauchy"}
        Robust loss name.
    f_scale : float
        Scale parameter for the robust loss.
    eps : float, optional
        Small positive value guarding against division by zero.
    """
    r = np.asarray(resid, dtype=float)
    fs = float(max(f_scale, eps))
    z = r / fs
    if loss == "linear":
        w = np.ones_like(z)
    elif loss == "soft_l1":
        w = (1.0 + z ** 2) ** -0.25
    elif loss == "huber":
        w = np.ones_like(z)
        mask = np.abs(z) > 1.0
        w[mask] = np.sqrt(1.0 / np.abs(z[mask]))
    elif loss == "cauchy":
        w = (1.0 + z ** 2) ** -0.5
    else:  # pragma: no cover - unknown loss
        raise ValueError(f"unknown loss '{loss}'")
    return w


def noise_weights(y, mode, floor=1e-12):
    """Return per-point noise weights according to ``mode``.

    Parameters
    ----------
    y : array_like
        Observed data.
    mode : {"none", "poisson", "inv_y"}
        Weighting strategy.
    floor : float, optional
        Minimum absolute value used when ``mode`` is ``inv_y``.
    """
    arr = np.asarray(y, dtype=float)
    if mode == "none":
        return np.ones_like(arr)
    if mode == "poisson":
        return 1.0 / np.sqrt(np.clip(np.abs(arr), 1.0, None))
    if mode == "inv_y":
        return 1.0 / np.clip(np.abs(arr), floor, None)
    raise ValueError(f"unknown mode '{mode}'")
