import numpy as np
import pytest

from core.uncertainty import bayesian_ci


def _dummy_predict(x):
    def f(th):
        return th[0] * x + th[1]
    return f


@pytest.mark.skipif(pytest.importorskip("emcee", reason=None) is None, reason="emcee required")
def test_bayes_band_gating():
    x = np.linspace(0.0, 1.0, 64)
    y = np.zeros_like(x)
    theta = np.array([0.1, 1.0, 0.5, 0.25], float)

    out_ok = bayesian_ci(
        theta_hat=theta,
        predict_full=_dummy_predict(x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - _dummy_predict(x)(th),
        n_walkers=12,
        n_burn=50,
        n_steps=60,
        thin=1,
        fit_ctx={
            "bayes_diagnostics": True,
            "bayes_band_enabled": True,
            "bayes_diag_ess_min": 0.0,
            "bayes_diag_rhat_max": 10.0,
            "bayes_diag_mcse_mean": float("inf"),
        },
        return_band=True,
        seed=123,
    )
    assert out_ok.band is not None

    out_fail = bayesian_ci(
        theta_hat=theta,
        predict_full=_dummy_predict(x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - _dummy_predict(x)(th),
        n_walkers=12,
        n_burn=50,
        n_steps=60,
        thin=1,
        fit_ctx={
            "bayes_diagnostics": True,
            "bayes_band_enabled": True,
            "bayes_diag_ess_min": 1e6,
            "bayes_diag_rhat_max": 1.0,
            "bayes_diag_mcse_mean": 0.0,
        },
        return_band=True,
        seed=123,
    )
    assert out_fail.band is None
    assert bool(out_fail.diagnostics.get("band_gated"))

    out_forced = bayesian_ci(
        theta_hat=theta,
        predict_full=_dummy_predict(x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - _dummy_predict(x)(th),
        n_walkers=12,
        n_burn=50,
        n_steps=60,
        thin=1,
        fit_ctx={
            "bayes_diagnostics": True,
            "bayes_band_enabled": True,
            "bayes_band_force": True,
            "bayes_diag_ess_min": 1e6,
            "bayes_diag_rhat_max": 1.0,
            "bayes_diag_mcse_mean": 0.0,
        },
        return_band=True,
        seed=123,
    )
    assert out_forced.band is not None
    assert not bool(out_forced.diagnostics.get("band_gated"))
    assert bool(out_forced.diagnostics.get("band_forced"))
