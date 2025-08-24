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
