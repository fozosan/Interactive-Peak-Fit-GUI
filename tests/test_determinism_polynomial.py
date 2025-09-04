import numpy as np
from core import models, peaks, fit_api


def test_determinism_polynomial():
    x = np.linspace(-3, 3, 121)
    seeds = [peaks.Peak(-0.5, 1.0, 0.6, 0.3), peaks.Peak(0.7, 0.8, 0.5, 0.4)]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "perf_seed_all": True,
        "baseline": {"method": "polynomial", "degree": 2, "normalize_x": True},
        "baseline_uses_fit_range": True,
    }
    res1 = fit_api.run_fit_consistent(x, y, seeds, cfg, None, "add", mask, rng_seed=1)
    res2 = fit_api.run_fit_consistent(x, y, seeds, cfg, None, "add", mask, rng_seed=1)
    assert np.allclose(res1["baseline"], res2["baseline"])
    assert np.allclose(res1["theta"], res2["theta"])
