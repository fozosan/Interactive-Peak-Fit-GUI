import numpy as np
from core import fit_api, uncertainty


def test_unc_bootstrap_outputs(two_peak_data):
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    res1 = uncertainty.bootstrap_ci(fit, n_boot=200, seed=42, workers=0)
    stats = res1["param_stats"]
    assert np.any(stats["std"] > 0)
    band = res1["band"]
    x, lo, hi = band["x"], band["lo"], band["hi"]
    assert len(x) == len(lo) == len(hi)
    res2 = uncertainty.bootstrap_ci(fit, n_boot=200, seed=42, workers=0)
    assert np.allclose(res1["param_mean"], res2["param_mean"])
    assert np.allclose(res1["param_std"], res2["param_std"])
