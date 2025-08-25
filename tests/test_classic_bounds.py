import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from core.bounds_classic import make_bounds_classic


def test_classic_simple_bounds():
    x = np.linspace(0, 100, 801)
    y = np.sin(x*0.0)  # zeros
    peaks = [Peak(10, 1, 3, 0.5), Peak(90, 2, 6, 0.4)]
    p0, (lo,hi), struct = make_bounds_classic(x, y, peaks, fit_window=(0,100))
    assert np.all(np.isfinite(p0))
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(hi >= lo)
    # heights >= 0; widths >= fwhm_lo; centers within window
    for s in struct:
        assert p0[s['ih']] >= 0.0
        if s.get('ic') is not None:
            c = p0[s['ic']]
            assert 0.0 <= c <= 100.0
        if s.get('iw') is not None:
            assert p0[s['iw']] >= 2*np.median(np.diff(x))


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
