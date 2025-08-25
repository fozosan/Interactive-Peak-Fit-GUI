import numpy as np
import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from core.bounds_classic import make_bounds_classic


def test_classic_simple_bounds():
    x = np.linspace(0, 100, 801)
    y = np.zeros_like(x)
    peaks = [Peak(10, 1, 3, 0.5), Peak(90, 2, 6, 0.4)]
    lo, hi, wmin = make_bounds_classic(x, y, peaks, centers_in_window=True)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(hi >= lo)
    # widths lower bound equals returned wmin
    assert np.all(lo[::3] == 0.0)  # heights
    assert np.all(lo[-1] == wmin)


def test_no_far_away_peaks_after_fit():
    from fit import classic
    x = np.linspace(0, 100, 801)
    true = [Peak(30, 5, 5, 0.5), Peak(70, 3, 7, 0.4)]
    y = pv_sum(x, true)
    start = [Peak(28, 4, 6, 0.5), Peak(68, 2, 8, 0.4)]
    res = classic.solve(x, y, start, mode='add', baseline=None, opts={})
    assert res['success']
    # Fitted centers stay in window
    for pk in res['peaks']:
        assert x.min() <= pk.center <= x.max()


def test_classic_bounds_respected():
    from fit import classic
    x = np.linspace(0, 100, 801)
    true = [Peak(50, 5, 5, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(20, -3, 200, 0.5)]
    lo, hi, wmin = make_bounds_classic(x, y, start, centers_in_window=True)
    res = classic.solve(x, y, start, mode='add', baseline=None, opts={})
    pk = res['peaks'][0]
    assert 0.0 <= pk.height <= hi[0]
    assert wmin <= pk.fwhm <= hi[-1]
    st = classic.prepare_state(x, y, start, mode='add', baseline=None, opts={})['state']
    st, ok, c0, c1, info = classic.iterate(st)
    pk2 = st['peaks'][0]
    assert 0.0 <= pk2.height <= hi[0]
    assert wmin <= pk2.fwhm <= hi[-1]
