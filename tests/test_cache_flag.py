import os
import sys
import pathlib
import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from infra import performance

pytestmark = pytest.mark.filterwarnings("ignore:.*")


def test_cache_flag_controls_calls():
    calls = []

    def fake_als(y, lam=1, p=0.5, niter=10, tol=1e-3):
        calls.append(1)
        return np.zeros_like(y)

    data = np.ones(5)
    cache = {}

    def compute():
        key = 0 if performance.cache_baseline_enabled() else None
        if key is not None and key in cache:
            return cache[key]
        res = fake_als(data)
        if key is not None:
            cache[key] = res
        return res

    performance.set_cache_baseline(True)
    compute()
    compute()
    assert len(calls) == 1

    calls.clear()
    cache.clear()
    performance.set_cache_baseline(False)
    compute()
    compute()
    assert len(calls) == 2

    performance.set_cache_baseline(True)

