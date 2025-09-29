import numpy as np
import pytest

from core.uncertainty import bayesian_ci


@pytest.mark.skipif(pytest.importorskip("emcee", reason=None) is None, reason="emcee required")
def test_bayes_band_subset_deterministic():
    x = np.linspace(0.0, 1.0, 32)
    y = np.zeros_like(x)
    theta = np.array([0.2, 1.0, 0.5, 0.5], float)

    kwargs = dict(
        theta_hat=theta,
        predict_full=lambda th: th[0] * x + th[1],
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - (th[0] * x + th[1]),
        n_walkers=12,
        n_burn=40,
        n_steps=50,
        thin=1,
        fit_ctx={
            "bayes_diagnostics": True,
            "bayes_band_enabled": True,
            "bayes_band_max_draws": 16,
            "bayes_diag_ess_min": 0.0,
            "bayes_diag_rhat_max": 10.0,
            "bayes_diag_mcse_mean": float("inf"),
        },
        return_band=True,
        seed=777,
    )

    res1 = bayesian_ci(**kwargs)
    res2 = bayesian_ci(**kwargs)

    assert res1.band is not None and res2.band is not None
    xb1, lo1, hi1 = res1.band
    xb2, lo2, hi2 = res2.band

    np.testing.assert_allclose(xb1, xb2)
    np.testing.assert_allclose(lo1, lo2)
    np.testing.assert_allclose(hi1, hi2)
