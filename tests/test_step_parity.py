import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic, modern_vp, modern


def test_classic_step_matches_fit():
    x = np.linspace(0, 60, 400)
    true = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
    y = pv_sum(x, true)
    start = [Peak(19, 4, 6, 0.5), Peak(41, 1.5, 7, 0.3)]
    full = classic.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {})
    s = classic.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(20):
        s, ok, c0, c1, info = classic.iterate(s)
        assert info["backtracks"] <= 8
        if ok:
            assert c1 < c0
    peaks_final = s["peaks"]
    model = pv_sum(x, peaks_final)
    rmse = np.sqrt(np.mean((model - y) ** 2))
    assert rmse <= max(1.02 * full["rmse"], 1e-12)


def _run_modern_backend(backend):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 60, 400)
    tgt = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
    y = pv_sum(x, tgt) + 0.01 * rng.standard_normal(x.size)
    start = [Peak(19, 4, 6, 0.5), Peak(41, 1.5, 7, 0.3)]
    full_res = backend.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {})
    theta_full = full_res["theta"]
    peaks_full = [Peak(theta_full[4*i], theta_full[4*i+1], theta_full[4*i+2], theta_full[4*i+3]) for i in range(len(start))]
    rmse_full = np.sqrt(np.mean((pv_sum(x, peaks_full) - y) ** 2))
    s = backend.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(15):
        s, ok, c0, c1, info = backend.iterate(s)
        assert info["backtracks"] <= 8
        assert np.isfinite(c1)
        if ok:
            assert c1 < c0
    theta = s.get("theta")
    peaks_final = [Peak(theta[4*i], theta[4*i+1], theta[4*i+2], theta[4*i+3]) for i in range(len(start))]
    model = pv_sum(x, peaks_final)
    rmse = np.sqrt(np.mean((model - y) ** 2))
    assert rmse <= 1.03 * rmse_full


def test_modern_vp_step():
    _run_modern_backend(modern_vp)


def test_modern_trf_step():
    _run_modern_backend(modern)
