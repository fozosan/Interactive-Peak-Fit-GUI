import numpy as np
from core import signals
from infra import performance


def compute_cached(x, y, deg, norm, mask, cache):
    key = None
    if performance.cache_baseline_enabled():
        slice_key = None
        if mask is not None and np.any(mask):
            slice_key = (float(x[mask][0]), float(x[mask][-1]))
        key = (hash(y.tobytes()), "poly", int(deg), bool(norm), slice_key)
    if key is not None and key in cache:
        return cache[key]
    base = signals.polynomial_baseline(
        x, y, degree=deg, mask=mask, normalize_x=norm
    )
    if key is not None:
        cache[key] = base
    return base


def test_poly_baseline_caching(monkeypatch):
    x = np.linspace(-5, 5, 101)
    y = x ** 2
    cache = {}
    calls = {"n": 0}

    orig = signals.polynomial_baseline

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(signals, "polynomial_baseline", wrapped)
    performance.set_cache_baseline(True)

    compute_cached(x, y, 2, True, None, cache)
    assert calls["n"] == 1
    compute_cached(x, y, 2, True, None, cache)
    assert calls["n"] == 1

    compute_cached(x, y, 3, True, None, cache)
    assert calls["n"] == 2

    compute_cached(x, y, 3, False, None, cache)
    assert calls["n"] == 3

    mask1 = (x > -1) & (x < 1)
    compute_cached(x, y, 3, False, mask1, cache)
    assert calls["n"] == 4
    compute_cached(x, y, 3, False, mask1, cache)
    assert calls["n"] == 4

    mask2 = (x > -2) & (x < 2)
    compute_cached(x, y, 3, False, mask2, cache)
    assert calls["n"] == 5
