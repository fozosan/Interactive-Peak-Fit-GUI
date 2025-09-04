import numpy as np
import pytest

from core.uncertainty import bootstrap_ci, bayesian_ci


def test_bootstrap_band_graceful_without_model():
    theta = np.array([1.0, 5.0, 2.0, 0.5])
    r = np.zeros(100)
    J = np.zeros((100, theta.size))
    x = np.linspace(0, 1, 100)
    y = np.zeros_like(x)
    res = bootstrap_ci(
        theta=theta,
        residual=r,
        jacobian=J,
        predict_full=None,
        x_all=x,
        y_all=y,
        fit_ctx={"refit": lambda th, *a: th},
        return_band=True,
        n_boot=10,
    )
    assert res.get("band") is None
    diag = res.diagnostics
    assert diag.get("band_source") is None
    assert diag.get("band_reason") is not None


def test_bayes_band_graceful_without_model():
    pytest.importorskip("emcee")
    theta = np.array([1.0, 5.0, 2.0, 0.5])
    x = np.linspace(0, 1, 100)
    y = np.zeros_like(x)
    res = bayesian_ci(
        theta_hat=theta,
        model=None,
        residual_fn=lambda th: np.ones_like(x) * 0.1,
        x_all=x,
        y_all=y,
        return_band=True,
        n_steps=10,
        n_burn=5,
    )
    assert res.get("band") is None

