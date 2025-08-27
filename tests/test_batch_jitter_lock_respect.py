from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import models, peaks, fit_api


def test_batch_jitter_lock_respect():
    x = np.linspace(-5, 5, 201)
    seeds = [
        peaks.Peak(-1.0, 1.0, 0.8, 0.3, lock_center=True, lock_width=True),
        peaks.Peak(1.0, 0.5, 0.6, 0.2),
    ]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_maxfev": 100,
        "solver_restarts": 3,
        "solver_jitter_pct": 10,
        "perf_seed_all": True,
    }
    res = fit_api.run_fit_consistent(x, y, seeds, cfg, None, "subtract", mask, rng_seed=0)
    theta = res["theta"]
    assert abs(theta[0] - seeds[0].center) <= 1e-12
    assert abs(theta[2] - seeds[0].fwhm) <= 1e-12

