"""MoS2-like Raman spectra for the uncertainty tests.

The numbers come from the 2026-11-07 uncertainty review of the user's 800g
MoS2 spectra: the x axis spans -133 to 1069 cm-1 in 1650 points, the detector
bias is 502.5 counts, the background sits about 2.4 counts above it, and the
noise variance is 0.724 + 0.0451 * S counts^2, where S is the signal above the
bias. The two lines are E2g and A1g of the 5 s spectrum, with eta 0.5, which
every solver holds fixed.
"""

from __future__ import annotations

import numpy as np

from core.models import pseudo_voigt

BIAS = 502.5
BACKGROUND = 504.9
NOISE_FLOOR = 0.724
NOISE_GAIN = 0.0451
# (center, height, fwhm, eta) of E2g and A1g
MOS2_LINES = ((382.92, 68.7, 2.32, 0.5), (408.25, 92.3, 2.75, 0.5))
WINDOW = (340.0, 440.0)
# Template seeds as a user would place them: 383 and 408, a little off.
TEMPLATE = [
    {"center": 383.0, "height": 60.0, "fwhm": 3.0, "eta": 0.5},
    {"center": 408.0, "height": 80.0, "fwhm": 3.0, "eta": 0.5},
]


def mos2_x() -> np.ndarray:
    return np.linspace(-132.994, 1068.86, 1650)


def mos2_mean(x: np.ndarray, lines=MOS2_LINES) -> np.ndarray:
    mu = np.full_like(x, BACKGROUND, dtype=float)
    for c, h, w, e in lines:
        mu = mu + pseudo_voigt(x, h, c, w, e)
    return mu


def mos2_spectrum(seed: int, lines=MOS2_LINES):
    """One noisy spectrum: (x, y) with the measured count noise."""
    x = mos2_x()
    mu = mos2_mean(x, lines)
    var = NOISE_FLOOR + NOISE_GAIN * np.maximum(mu - BIAS, 0.0)
    rng = np.random.default_rng(seed)
    return x, mu + np.sqrt(var) * rng.standard_normal(x.size)


def write_spectrum(path, x, y) -> None:
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")
