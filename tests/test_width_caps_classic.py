import numpy as np
from core.models import pv_sum
from core.peaks import Peak
from fit import classic


def test_classic_solver_honors_width_cap():
    x = np.linspace(0, 100, 801)
    true = [Peak(50.0, 5.0, 10.0, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(45.0, 4.0, 20.0, 0.5)]  # start wider than cap
    res = classic.solve(
        x, y, start, mode="add", baseline=None,
        opts={"centers_in_window": True, "maxfev": 2000, "width_caps": [6.0]},
    )
    assert res["success"]
    fwhm = res["peaks"][0].fwhm
    assert fwhm <= 6.0 + 1e-6
