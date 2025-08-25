import numpy as np


def noise_weights(y, mode):
    r"""Return element-wise noise weights for ``y``.

    ``mode`` may be ``"none"`` (``None`` is returned), ``"poisson```` for
    :math:`1/\sqrt{|y|}` or ``"inv_y"`` for :math:`1/|y|`.  Any non-finite
    results are set to ``0`` to keep subsequent calculations stable.
    """

    if mode == "none":
        return None
    y = np.asarray(y, float)
    if mode == "poisson":
        w = 1.0 / np.sqrt(np.clip(np.abs(y), 1e-12, None))
    elif mode == "inv_y":
        w = 1.0 / np.clip(np.abs(y), 1e-12, None)
    else:
        return None
    w[~np.isfinite(w)] = 0.0
    return w


def robust_weights(resid, loss, f_scale=1.0):
    """Return IRLS weights for ``resid`` under ``loss``.

    ``loss`` follows the :func:`scipy.optimize.least_squares` names and
    ``f_scale`` provides the scaling.  ``None`` is returned for linear loss.
    """

    if loss in (None, "", "linear"):
        return None
    r = np.asarray(resid, float) / max(1e-12, float(f_scale))
    a = np.abs(r)
    if loss == "soft_l1":
        w = 1.0 / np.sqrt(1.0 + a * a)
    elif loss == "huber":
        k = 1.0
        w = np.where(a <= k, 1.0, k / a)
    elif loss == "cauchy":
        w = 1.0 / (1.0 + a * a)
    else:
        return None
    w[~np.isfinite(w)] = 0.0
    return w


def combine_weights(wn, wr):
    """Combine noise (``wn``) and robust (``wr``) weights.

    ``None`` inputs are treated as unity.  ``None`` is returned only if both are
    ``None``; otherwise the element-wise product clipped to finite values is
    returned.
    """

    if wn is None and wr is None:
        return None
    if wn is None:
        w = wr
    elif wr is None:
        w = wn
    else:
        w = wn * wr
    w = np.asarray(w, float)
    w[~np.isfinite(w)] = 0.0
    return w
