import numpy as np
from core.signals import polynomial_baseline


def test_poly_baseline_fit_range_matches_polyfit():
    x = np.linspace(-1, 1, 101)
    y = 1.0 + 2.0 * x
    mask = np.abs(x) <= 0.2  # center 40%
    baseline = polynomial_baseline(x, y, degree=1, mask=mask, normalize_x=True)
    coef = np.polyfit(x[mask], y[mask], 1)
    truth = np.polyval(coef, x)
    assert np.allclose(baseline, truth, rtol=1e-6, atol=1e-6)
