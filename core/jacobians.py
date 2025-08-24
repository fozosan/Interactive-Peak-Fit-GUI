import numpy as np


def pv_and_grads(x, h, c, f, eta):
    """Evaluate a pseudo-Voigt peak and its gradients.

    Parameters
    ----------
    x : array_like
        Sample positions.
    h, c, f, eta : float
        Peak height, centre, FWHM and mixing parameter.

    Returns
    -------
    pv : ndarray
        Peak values at ``x``.
    d_pv_dc : ndarray
        Gradient with respect to the centre.
    d_pv_df : ndarray
        Gradient with respect to the FWHM.
    """
    x = np.asarray(x, dtype=float)
    f = float(max(f, 1e-12))
    dx = x - c
    A = 4.0 * np.log(2.0)
    B = 4.0
    g = np.exp(-A * (dx ** 2) / f ** 2)
    dg_dc = g * (2.0 * A) * dx / f ** 2
    dg_df = g * (2.0 * A) * (dx ** 2) / f ** 3
    denom = 1.0 + B * (dx ** 2) / f ** 2
    L = 1.0 / denom
    dL_dc = (2.0 * B) * dx / f ** 2 / denom ** 2
    dL_df = (2.0 * B) * (dx ** 2) / f ** 3 / denom ** 2
    base = (1.0 - eta) * g + eta * L
    pv = h * base
    d_pv_dc = h * ((1.0 - eta) * dg_dc + eta * dL_dc)
    d_pv_df = h * ((1.0 - eta) * dg_df + eta * dL_df)
    return pv, d_pv_dc, d_pv_df
