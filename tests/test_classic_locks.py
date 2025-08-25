import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import classic


def test_classic_locked_params_immutable():
    x = np.linspace(0, 60, 400)
    true = [Peak(20, 5, 5, 0.5), Peak(40, 3, 6, 0.3)]
    y = pv_sum(x, true)
    start = [
        Peak(25, 4, 5, 0.5, lock_center=True),
        Peak(38, 2, 8, 0.3, lock_width=True),
    ]
    res = classic.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width) for p in start], "add", None, {})
    for s, pk in zip(start, res["peaks"]):
        if s.lock_center:
            assert abs(pk.center - s.center) < 1e-9
        if s.lock_width:
            assert abs(pk.fwhm - s.fwhm) < 1e-9
    prep = classic.prepare_state(x, y, [Peak(p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width) for p in start], "add", None, {})
    st = prep["state"]
    for _ in range(10):
        st, ok, c0, c1, info = classic.iterate(st)
        assert info["backtracks"] <= 10
        if ok:
            assert c1 <= c0
    for s, pk in zip(start, st["peaks"]):
        if s.lock_center:
            assert abs(pk.center - s.center) < 1e-9
        if s.lock_width:
            assert abs(pk.fwhm - s.fwhm) < 1e-9
