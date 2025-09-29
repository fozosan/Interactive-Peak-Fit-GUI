import numpy as np
import pytest

from core.uncertainty import bayesian_ci


pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _fixture(n=64, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    a_true, b_true = 1.2, 0.7
    y = a_true + b_true * x + rng.normal(scale=sigma, size=n)
    X = np.column_stack([np.ones_like(x), x])
    th_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    th_hat = np.asarray(th_hat, float)

    def model(theta):
        theta = np.asarray(theta, float)
        return theta[0] + theta[1] * x

    return x, y, th_hat, model


@pytest.mark.skipif(pytest.importorskip("emcee") is None, reason="emcee not available")
def test_bayes_band_gates_when_diagnostics_unhealthy():
    x, y, theta_hat, model = _fixture()

    fit_ctx = {
        "alpha": 0.1,
        "bayes_diagnostics": True,
        "bayes_band_enabled": True,
        "bayes_band_force": False,
        "bayes_band_max_draws": 64,
        # Intentionally impossible thresholds to force gating:
        "bayes_diag_ess_min": 1e9,
        "bayes_diag_rhat_max": 1e-6,
        "bayes_diag_mcse_mean": 0.0,
        "unc_band_workers": 0,
    }
    res = bayesian_ci(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        fit_ctx=fit_ctx,
        n_walkers=16,
        n_burn=100,
        n_steps=160,
        thin=2,
        seed=42,
        workers=None,
        return_band=True,
    )

    d = res.diagnostics or {}
    assert d.get("diagnostics_enabled") is True
    assert d.get("band_gated") is True
    # Threshold gating should report this reason:
    assert d.get("band_skip_reason") == "diagnostics_unhealthy"
    assert res.band is None


@pytest.mark.skipif(pytest.importorskip("emcee") is None, reason="emcee not available")
def test_bayes_band_forces_even_if_unhealthy():
    x, y, theta_hat, model = _fixture()

    fit_ctx = {
        "alpha": 0.1,
        "bayes_diagnostics": True,
        "bayes_band_enabled": True,
        "bayes_band_force": True,   # force it this time
        "bayes_band_max_draws": 64,
        "bayes_diag_ess_min": 1e9,  # still impossible, but should be ignored
        "bayes_diag_rhat_max": 1e-6,
        "bayes_diag_mcse_mean": 0.0,
        "unc_band_workers": 0,
    }
    res = bayesian_ci(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        fit_ctx=fit_ctx,
        n_walkers=16,
        n_burn=80,
        n_steps=160,
        thin=2,
        seed=43,
        workers=None,
        return_band=True,
    )

    d = res.diagnostics or {}
    assert d.get("band_forced") is True
    assert d.get("band_gated") is False
    assert d.get("band_source") == "posterior_predictive_subset"
    assert d.get("band_draws_used", 0) > 0
    assert res.band is not None
