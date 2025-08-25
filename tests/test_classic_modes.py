import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import classic


def test_classic_solve_add_and_subtract():
    x = np.linspace(-5, 5, 400)
    baseline = 0.1 * x + 1.0
    true = [Peak(-1.0, 2.0, 1.2, 0.3), Peak(2.0, 1.5, 0.8, 0.6)]
    y_raw = pv_sum(x, true) + baseline
    start = [
        Peak(-1.0, 1.8, 1.0, 0.3, lock_center=True),
        Peak(2.2, 1.0, 0.8, 0.6, lock_width=True),
    ]
    res_add = classic.solve(x, y_raw, start, mode="add", baseline=baseline, opts={})
    assert res_add["success"]
    assert res_add["rmse"] < 1e-3
    res_sub = classic.solve(x, y_raw, start, mode="subtract", baseline=baseline, opts={})
    assert res_sub["success"]
    assert res_sub["rmse"] < 1e-3
