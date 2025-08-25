import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import classic

def test_classic_physical_clamps():
    x = np.linspace(0, 100, 801)
    true = [Peak(50, 5, 5, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(48, -3, 1e-9, 0.5)]
    res = classic.solve(x, y, start, mode="add", baseline=None, opts={})
    pk = res["peaks"][0]
    assert pk.height >= 0.0
    assert pk.fwhm >= 1e-6
    st = classic.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(5):
        st, ok, c0, c1, info = classic.iterate(st)
        pk2 = st["peaks"][0]
        assert pk2.height >= 0.0
        assert pk2.fwhm >= 1e-6
