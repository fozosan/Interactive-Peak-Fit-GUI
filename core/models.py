"""Spectral peak models for Peakfit 3.x.

Only pseudo-Voigt support is sketched out here. The module is designed to
switch between ``numpy`` and ``cupy`` via an ``xp`` alias.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:  # optional CuPy support
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy may be absent
    cp = None  # type: ignore

# alias; will be set to ``cp`` when GPU mode is enabled
xp = np

__all__ = [
    "pv_sum",
    "pv_sum_with_jac",
    "pv_design_matrix",
    "pv_area",
    "pseudo_voigt",
]


def pseudo_voigt(
    x,
    height: float,
    x0: float,
    fwhm: float,
    eta: float,
    xp_module: Optional[object] = None,
):
    """
    Canonical pseudo-Voigt: eta*L + (1-eta)*G, parameterized by FWHM.
    - x: array-like (xp ndarray), same xp as module-level 'xp'
    - xp_module: optional array module override; defaults to module 'xp'
    - Clips eta to [0,1]; requires fwhm>0
    Returns array same shape/dtype as x.
    """

    xp_local = xp_module or xp  # use module alias already present
    if fwhm <= 0:
        raise ValueError("pseudo_voigt: fwhm must be > 0")

    x_arr = xp_local.asarray(x, dtype=float)
    eta_c = xp_local.clip(eta, 0.0, 1.0)

    sigma = fwhm / (2.0 * xp_local.sqrt(2.0 * xp_local.log(2.0)))
    gamma = fwhm / 2.0
    dx = x_arr - x0

    gauss = xp_local.exp(-(dx * dx) / (2.0 * sigma * sigma))
    lor = (gamma * gamma) / (dx * dx + gamma * gamma)
    y = height * (eta_c * lor + (1.0 - eta_c) * gauss)
    return y


def pv_sum(x: np.ndarray, peaks: list, xp_module=xp) -> np.ndarray:
    """Return the sum of pseudo-Voigt peaks evaluated at ``x``.

    ``peaks`` should be an iterable of objects with ``center``, ``height``,
    ``fwhm`` and ``eta`` attributes (e.g. :class:`core.peaks.Peak`). The
    implementation is vectorized and works for either NumPy or CuPy arrays via
    the ``xp_module`` parameter.
    """

    x = xp_module.asarray(x, dtype=float)
    y = xp_module.zeros_like(x)
    for p in peaks:
        dx = (x - p.center) / p.fwhm
        gaussian = xp_module.exp(-4.0 * xp_module.log(2.0) * dx**2)
        lorentz = 1.0 / (1.0 + 4.0 * dx**2)
        y += p.height * ((1.0 - p.eta) * gaussian + p.eta * lorentz)
    return y


def pv_sum_with_jac(
    x: np.ndarray,
    centers: np.ndarray,
    heights: np.ndarray,
    fwhm: np.ndarray,
    eta: np.ndarray,
    xp_module=xp,
):
    """Return ``y`` and analytic Jacobian for pseudo-Voigt peaks.

    Parameters are arrays of per-peak values.  The Jacobian is dense with
    derivative columns ordered as ``[dh, dc, df]`` for each peak.  Widths are
    internally clipped to ``>=1e-6`` to avoid divisions by zero.
    """

    x = xp_module.asarray(x, dtype=float)
    c = xp_module.asarray(centers, dtype=float)
    h = xp_module.asarray(heights, dtype=float)
    f = xp_module.maximum(xp_module.asarray(fwhm, dtype=float), 1e-6)
    e = xp_module.asarray(eta, dtype=float)

    dx = x[:, None] - c[None, :]

    A = 4.0 * xp_module.log(2.0)
    B = 4.0

    g = xp_module.exp(-A * (dx**2) / f**2)
    dg_dc = g * (2.0 * A) * dx / f**2
    dg_df = g * (2.0 * A) * (dx**2) / f**3

    denom = 1.0 + B * (dx**2) / f**2
    L = 1.0 / denom
    dL_dc = (2.0 * B) * dx / f**2 / denom**2
    dL_df = (2.0 * B) * (dx**2) / f**3 / denom**2

    e = e[None, :]
    base = (1.0 - e) * g + e * L
    y = (h[None, :] * base).sum(axis=1)

    dh = base
    dc = h[None, :] * ((1.0 - e) * dg_dc + e * dL_dc)
    df = h[None, :] * ((1.0 - e) * dg_df + e * dL_df)

    return y, dh, dc, df


def pv_design_matrix(x: np.ndarray, peaks: list, xp_module=xp) -> np.ndarray:
    """Return design matrix for height-only fits.

    Each column ``k`` is the pseudo-Voigt peak ``k`` evaluated at unit height
    with fixed centre/FWHM/eta from ``peaks``.
    """

    centers = [p.center for p in peaks]
    widths = [p.fwhm for p in peaks]
    etas = [p.eta for p in peaks]
    heights = [1.0] * len(peaks)
    _, dh, _, _ = pv_sum_with_jac(
        x, np.asarray(centers), np.asarray(heights), np.asarray(widths), np.asarray(etas), xp_module
    )
    return xp_module.asarray(dh)

def pv_area(height: float, fwhm: float, eta: float) -> float:
    """Return the analytic area of a pseudo-Voigt peak."""

    gauss_area = height * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    lorentz_area = height * fwhm * np.pi / 2.0
    return (1.0 - eta) * gauss_area + eta * lorentz_area
