"""Jittered restarts must respect parameter locks."""

import numpy as np

from core import fit_api, models, peaks


def test_jitter_respects_locks():
    x = np.linspace(-5, 5, 201)
    true_peaks = [
        peaks.Peak(-1.0, 1.0, 0.8, 0.3),
        peaks.Peak(1.0, 0.7, 0.6, 0.2),
    ]
    y = models.pv_sum(x, true_peaks)
    seeds = [
        peaks.Peak(-1.0, 1.0, 0.8, 0.3, lock_center=True, lock_width=True),
        peaks.Peak(1.0, 0.5, 0.6, 0.2),
    ]
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_restarts": 3,
        "solver_jitter_pct": 10,
        "modern_vp": {"min_fwhm": 0.5, "max_fwhm": 1.5, "bound_centers_to_window": True},
        "perf_seed_all": True,
    }

    res = fit_api.run_fit_consistent(x, y, seeds, cfg, None, "add", mask, rng_seed=0)
    theta = res["theta"]

    assert abs(theta[0] - seeds[0].center) <= 1e-12
    assert abs(theta[2] - seeds[0].fwhm) <= 1e-12

    lo, hi = res["bounds"]
    assert np.all(theta >= lo - 1e-12)
    assert np.all(theta <= hi + 1e-12)

    moved = abs(theta[5] - seeds[1].height) > 1e-6 or abs(theta[4] - seeds[1].center) > 1e-6
    assert moved

