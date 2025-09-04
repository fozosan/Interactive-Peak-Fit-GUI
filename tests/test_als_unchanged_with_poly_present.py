import numpy as np
from core import models, peaks, signals, fit_api


def test_als_unchanged_with_poly_present():
    x = np.linspace(-5, 5, 201)
    seeds = [peaks.Peak(0.0, 1.0, 1.0, 0.5)]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "baseline_uses_fit_range": True,
    }
    baseline = signals.als_baseline(y, lam=1e5, p=0.001, niter=10, tol=0.0)
    res_manual = fit_api.run_fit_consistent(x, y, seeds, cfg, baseline, "add", mask)
    res_auto = fit_api.run_fit_consistent(x, y, seeds, cfg, None, "add", mask)
    assert np.allclose(res_manual["baseline"], baseline)
    assert np.allclose(res_auto["baseline"], baseline)
    assert np.allclose(res_manual["theta"], res_auto["theta"])
