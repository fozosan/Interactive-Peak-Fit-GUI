from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import models, peaks, fit_api


def test_batch_bounds_enforced():
    x = np.linspace(-4, 4, 101)
    seed = peaks.Peak(0.0, 1.0, 1.0, 0.5, lock_center=True, lock_width=True)
    y = models.pv_sum(x, [seed])
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_maxfev": 50,
        "solver_restarts": 3,
        "solver_jitter_pct": 20,
        "modern_vp": {"min_fwhm": 0.9, "max_fwhm": 1.1, "bound_centers_to_window": True},
    }
    res = fit_api.run_fit_consistent(x, y, [seed], cfg, None, "subtract", mask, rng_seed=0)
    theta = res["theta"]
    lo, hi = res["bounds"]
    assert np.all(theta >= lo - 1e-12)
    assert np.all(theta <= hi + 1e-12)
    assert abs(theta[0] - seed.center) <= 1e-12
    assert abs(theta[2] - seed.fwhm) <= 1e-12

