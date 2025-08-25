import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import modern_vp, modern

try:
    from fit import lmfit_backend
    import lmfit  # noqa: F401
    HAVE_LMFIT = True
except Exception:  # pragma: no cover - optional
    HAVE_LMFIT = False

x = np.linspace(0, 60, 400)
true_peaks = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
y = pv_sum(x, true_peaks)
start = [Peak(19, 4, 6, 0.5), Peak(41, 1.5, 7, 0.3)]

# Baselines captured from verified solver outputs
BASE_VP_COST = 3.2862852197406284e-14
BASE_VP_THETA = np.array([20.0, 5.0, 5.0, 0.5, 39.99999991, 2.0, 6.0, 0.3])

BASE_TRF_COST = 2.10383544799526e-27
BASE_TRF_THETA = np.array([20.0, 5.0, 5.0, 0.5, 40.0, 2.0, 6.0, 0.3])

BASE_LMFIT_COST = 3.5217440610974547e-29
BASE_LMFIT_THETA = np.array([20.0, 5.0, 5.0, 0.5, 40.0, 2.0, 6.0, 0.3])


def _relative_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def test_modern_vp_regression():
    res = modern_vp.solve(x, y, start, mode="add", baseline=None, options={})
    assert res["ok"]
    assert _relative_diff(res["cost"], BASE_VP_COST) < 1e-6
    assert np.allclose(res["theta"], BASE_VP_THETA, rtol=1e-6, atol=1e-8)


def test_modern_trf_regression():
    res = modern.solve(x, y, start, mode="add", baseline=None, options={})
    assert res["ok"]
    assert _relative_diff(res["cost"], BASE_TRF_COST) < 1e-6
    assert np.allclose(res["theta"], BASE_TRF_THETA, rtol=1e-6, atol=1e-8)


def test_lmfit_vp_regression():
    if not HAVE_LMFIT:
        import pytest

        pytest.skip("lmfit not installed")
    res = lmfit_backend.solve(x, y, start, mode="add", baseline=None, options={})
    assert res["ok"]
    assert _relative_diff(res["cost"], BASE_LMFIT_COST) < 1e-6
    assert np.allclose(res["theta"], BASE_LMFIT_THETA, rtol=1e-6, atol=1e-8)
