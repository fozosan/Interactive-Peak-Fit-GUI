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


# The weak lines of the 0.20 s spectrum (review: 9.8 and 12.5 counts), with the
# shapes of the 5 s spectrum, over a background 0.7 counts above the bias.
MOS2_WEAK_LINES = ((382.92, 9.8, 2.32, 0.5), (408.25, 12.5, 2.75, 0.5))
WEAK_BACKGROUND = 503.2
# The 1 s Si line (review), over a background 27.4 counts above the bias.
SI_LINES = ((520.02, 2142.5, 6.62, 0.5),)
SI_WINDOW = (490.0, 550.0)
SI_BACKGROUND = 529.9
SI_TEMPLATE = [{"center": 520.5, "height": 2000.0, "fwhm": 4.0, "eta": 0.5}]

# A realistic curved background, measured on the 5 s MoS2 spectrum (file 07):
# its lines at 383 and 408 cm-1 (eta 0.5) and a Legendre series of order 6 over
# 300-480 cm-1 fitted together, weighted by its noise model (PR E probe
# measure_shape.py). In the 340-440 window the series rises 1.7 counts and
# bends 0.64 counts away from a straight line and 0.31 from a parabola; a cubic
# follows it to 0.07. The Si window's background cannot be measured on the Si
# spectrum, whose pseudo-Voigt misfit is tens of counts, so the Si case maps the
# same shape onto 490-550 and scales it by the ratio of the noise floors, 1.54.
CURVED_SPAN = (300.0, 480.0)
CURVED_COEF = (504.8309, 1.0685, 0.0822, -0.5849, -0.9838, -1.0974, -0.4234)
SI_CURVE_SCALE = 1.54


def mos2_x() -> np.ndarray:
    return np.linspace(-132.994, 1068.86, 1650)


def curved_shape(x: np.ndarray, window=WINDOW, scale: float = 1.0) -> np.ndarray:
    """The measured background shape over ``window``, about its mean there, times ``scale``.

    The shape was measured over 340-440 cm-1; another window is mapped onto that
    one point for point, so the Si window gets the same curvature in its own span.
    Outside the window it stays at its edge values.
    """

    x = np.asarray(x, dtype=float)
    lo, hi = window
    xm = WINDOW[0] + (np.clip(x, lo, hi) - lo) * (WINDOW[1] - WINDOW[0]) / (hi - lo)
    u = 2.0 * (xm - CURVED_SPAN[0]) / (CURVED_SPAN[1] - CURVED_SPAN[0]) - 1.0
    inside = (x >= lo) & (x <= hi)
    s = np.polynomial.legendre.legval(u, CURVED_COEF)
    return scale * (s - s[inside].mean())


def mos2_mean(x: np.ndarray, lines=MOS2_LINES, background=None) -> np.ndarray:
    """The mean spectrum: ``background`` (default: flat at BACKGROUND) plus the lines."""
    if background is None:
        mu = np.full_like(x, BACKGROUND, dtype=float)
    else:
        mu = np.asarray(background, dtype=float).copy()
    for c, h, w, e in lines:
        mu = mu + pseudo_voigt(x, h, c, w, e)
    return mu


def mos2_spectrum(seed: int, lines=MOS2_LINES, background=None):
    """One noisy spectrum: (x, y) with the measured count noise."""
    x = mos2_x()
    mu = mos2_mean(x, lines, None if background is None else background(x))
    var = NOISE_FLOOR + NOISE_GAIN * np.maximum(mu - BIAS, 0.0)
    rng = np.random.default_rng(seed)
    return x, mu + np.sqrt(var) * rng.standard_normal(x.size)


def write_spectrum(path, x, y) -> None:
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")
