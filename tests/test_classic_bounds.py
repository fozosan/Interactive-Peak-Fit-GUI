import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import classic


def test_classic_centers_in_window():
    x = np.linspace(0, 100, 801)
    true = [Peak(30, 5, 5, 0.5), Peak(70, 3, 7, 0.4)]
    y = pv_sum(x, true)
    start = [Peak(10, 4, 6, 0.5), Peak(90, 2, 8, 0.4)]
    res = classic.solve(x, y, start, mode="add", baseline=None, opts={"centers_in_window": True})
    assert res["success"]
    for pk in res["peaks"]:
        assert x.min() <= pk.center <= x.max()


def test_classic_bounds_respected():
    x = np.linspace(0, 100, 801)
    true = [Peak(50, 5, 5, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(20, -3, 1e-9, 0.5)]
    res = classic.solve(x, y, start, mode="add", baseline=None, opts={"centers_in_window": True})
    pk = res["peaks"][0]
    assert pk.height >= 0.0
    assert pk.fwhm >= 1e-6
    assert x.min() <= pk.center <= x.max()
    st = classic.prepare_state(x, y, start, mode="add", baseline=None, opts={"centers_in_window": True})["state"]
    st, ok, c0, c1, info = classic.iterate(st)
    pk2 = st["peaks"][0]
    assert pk2.height >= 0.0
    assert pk2.fwhm >= 1e-6
    assert x.min() <= pk2.center <= x.max()
