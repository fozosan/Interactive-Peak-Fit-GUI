import numpy as np
from core import signals

def test_poly_degree_guard():
    x = np.linspace(0, 10, 5)
    y = x ** 2
    mask = np.zeros_like(x, dtype=bool)
    mask[:3] = True  # only three points available
    baseline, used = signals.polynomial_baseline(
        x,
        y,
        degree=5,
        mask=mask,
        normalize_x=True,
        return_used_degree=True,
    )
    assert used == mask.sum() - 1
    assert baseline.shape == x.shape
