import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.peaks import Peak
from fit import lmfit_backend


def test_lmfit_respects_locks_and_bounds():
    lmfit = pytest.importorskip("lmfit")
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    peaks = [Peak(5.0, -1.0, -0.5, 1.2, lock_center=True, lock_width=True)]
    res = lmfit_backend.solve(
        x,
        y,
        peaks,
        mode="add",
        baseline=None,
        options={"centers_in_window": True, "min_fwhm": 1.0},
    )
    theta = res["theta"]
    assert np.allclose(theta, [5.0, 0.0, 1.0, 1.0])


def test_clip_params_enforces_bounds():
    lmfit = pytest.importorskip("lmfit")
    params = lmfit.Parameters()
    params.add("a", value=-1, min=0, max=2)
    params.add("b", value=3, min=0, max=2)
    lmfit_backend._clip_params(params)
    assert params["a"].value == 0
    assert params["b"].value == 2


def test_lmfit_leastsq_stays_within_bounds():
    lmfit = pytest.importorskip("lmfit")
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    peaks = [Peak(5.0, 1.0, 2.0, 0.5)]
    res = lmfit_backend.solve(
        x,
        y,
        peaks,
        mode="add",
        baseline=None,
        options={"centers_in_window": True, "min_fwhm": 1.0, "algo": "leastsq"},
    )
    c, h, w, e = res["theta"]
    assert 0.0 <= c <= 10.0
    assert h >= 0.0
    assert w >= 1.0
    assert 0.0 <= e <= 1.0
