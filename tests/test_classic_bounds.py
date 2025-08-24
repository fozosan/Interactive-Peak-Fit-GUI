import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic


def test_classic_respects_bounds():
    x = np.linspace(0, 10, 200)
    peaks = [Peak(2.0, 1.0, 0.5, 0.5), Peak(5.0, 0.8, 0.4, 0.5)]
    y = pv_sum(x, peaks)
    res = classic.solve(x, y, peaks, "subtract", None, {})
    centers = res["theta"][0::4]
    widths = res["theta"][2::4]
    xmin, xmax = float(x.min()), float(x.max())
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(2.0 * dx, 1e-6)
    fwhm_max = 0.5 * (xmax - xmin)
    assert np.all(centers >= xmin) and np.all(centers <= xmax)
    assert np.all(widths >= fwhm_min) and np.all(widths <= fwhm_max)
    assert res["ok"] and np.isfinite(res["cost"])


def test_classic_lock_respected():
    x = np.linspace(0, 3, 100)
    peaks = [
        Peak(1.0, 1.0, 0.2, 0.5, lock_center=True, lock_width=True),
        Peak(2.0, 0.8, 0.3, 0.5),
    ]
    y = pv_sum(x, peaks)
    res = classic.solve(x, y, peaks, "subtract", None, {})
    theta = res["theta"]
    assert np.isclose(theta[0], 1.0)
    assert np.isclose(theta[2], 0.2)


def test_classic_clamps_bad_start():
    x = np.linspace(0, 1, 200)
    true = [Peak(0.3, 1.0, 0.05, 0.5), Peak(0.7, 0.8, 0.04, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(5.0, 1.0, 1.0, 0.5), Peak(-3.0, 0.5, 2.0, 0.5)]
    model0 = pv_sum(x, start)
    cost0 = 0.5 * float(np.dot(model0 - y, model0 - y))
    res = classic.solve(x, y, start, "subtract", None, {})
    theta = res["theta"]
    centers = theta[0::4]
    widths = theta[2::4]
    xmin, xmax = float(x.min()), float(x.max())
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(2.0 * dx, 1e-6)
    fwhm_max = 0.5 * (xmax - xmin)
    assert np.all(centers >= xmin) and np.all(centers <= xmax)
    assert np.all(widths >= fwhm_min) and np.all(widths <= fwhm_max)
    assert res["cost"] < cost0
