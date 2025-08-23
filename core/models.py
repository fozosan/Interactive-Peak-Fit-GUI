"""Spectral peak models for Peakfit 3.x.

Only pseudo-Voigt support is sketched out here. The module is designed to
switch between ``numpy`` and ``cupy`` via an ``xp`` alias.
"""
from __future__ import annotations

import numpy as np

try:  # optional CuPy support
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy may be absent
    cp = None  # type: ignore

# alias; will be set to ``cp`` when GPU mode is enabled
xp = np


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


def pv_area(height: float, fwhm: float, eta: float) -> float:
    """Return the analytic area of a pseudo-Voigt peak."""

    gauss_area = height * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    lorentz_area = height * fwhm * np.pi / 2.0
    return (1.0 - eta) * gauss_area + eta * lorentz_area
