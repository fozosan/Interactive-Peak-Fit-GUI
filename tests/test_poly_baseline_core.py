import numpy as np
from core.signals import polynomial_baseline


def test_quadratic_recovery():
    x = np.linspace(-1, 1, 201)
    truth = 0.5 + 1.2 * x + 0.3 * x**2
    baseline = polynomial_baseline(x, truth, degree=2, normalize_x=True)
    assert np.max(np.abs(baseline - truth)) < 1e-6


def test_degree_zero_one():
    x = np.linspace(-2, 2, 11)
    y0 = np.full_like(x, 2.0)
    assert np.allclose(polynomial_baseline(x, y0, degree=0), 2.0)
    y1 = 3.0 + 2.0 * x
    base1 = polynomial_baseline(x, y1, degree=1)
    assert np.allclose(base1, y1)
