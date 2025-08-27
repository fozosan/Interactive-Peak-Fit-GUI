import numpy as np
from core import fit_api, uncertainty


def test_bootstrap_basic(two_peak_data, rng):
    noisy = dict(two_peak_data)
    noisy["y"] = noisy["y"] + 0.01 * rng.standard_normal(noisy["x"].size)
    res1 = uncertainty.bootstrap_ci(
        engine=fit_api.run_fit_consistent,
        data=noisy,
        n=30,
        band_percentiles=(2.5, 97.5),
        workers=0,
        seed_root=123,
    )
    assert res1["band"] is not None
    assert np.all(np.isfinite(res1["param_std"]))
    assert np.all(res1["param_std"] >= 0)
    assert np.any(res1["param_std"] > 0)

    res2 = uncertainty.bootstrap_ci(
        engine=fit_api.run_fit_consistent,
        data=noisy,
        n=30,
        band_percentiles=(2.5, 97.5),
        workers=0,
        seed_root=123,
    )
    assert np.allclose(res1["param_mean"], res2["param_mean"])
    assert np.allclose(res1["param_std"], res2["param_std"])
