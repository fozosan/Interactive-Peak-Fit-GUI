import numpy as np
import pytest

emcee = pytest.importorskip("emcee")
from core.uncertainty import bayesian_ci


def _toy_model(th, n=16):
    # 1D param + log-sigma inferred internally; produce zero residual signal
    return np.zeros(n, dtype=float)


def _residual_fn(th, n=16):
    return np.zeros(n, dtype=float)


def test_bayes_diag_disabled_skips_metrics(monkeypatch):
    # Raise if diagnostics are attempted
    monkeypatch.setattr("core.mcmc_utils.ess_autocorr", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ESS called")))
    monkeypatch.setattr("core.mcmc_utils.rhat_split", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Rhat called")))

    theta = np.array([0.5])  # single param; sigma handled internally
    res = bayesian_ci(
        theta_hat=theta,
        model=_toy_model,
        predict_full=_toy_model,
        x_all=np.arange(16, dtype=float),
        y_all=np.zeros(16),
        residual_fn=_residual_fn,
        fit_ctx={"bayes_diag": False, "alpha": 0.05},
        n_walkers=16, n_burn=10, n_steps=40, thin=2,
        seed=123, workers=0, return_band=False,
    )
    d = res.diagnostics or {}
    assert d.get("diagnostics_enabled") is False
    assert d.get("ess_min") is None
    assert d.get("rhat_max") is None
    assert d.get("mcse") is None


def test_bayes_diag_enabled_populates_metrics(monkeypatch):
    # Provide harmless, fast stubs
    monkeypatch.setattr("core.mcmc_utils.ess_autocorr", lambda post: np.ones((1,1,2)))
    monkeypatch.setattr("core.mcmc_utils.rhat_split", lambda post: np.ones((1,1,2)))

    theta = np.array([0.5])
    res = bayesian_ci(
        theta_hat=theta,
        model=_toy_model,
        predict_full=_toy_model,
        x_all=np.arange(16, dtype=float),
        y_all=np.zeros(16),
        residual_fn=_residual_fn,
        fit_ctx={"bayes_diag": True, "alpha": 0.05},
        n_walkers=16, n_burn=10, n_steps=40, thin=2,
        seed=123, workers=0, return_band=False,
    )
    d = res.diagnostics or {}
    assert d.get("diagnostics_enabled") is True
    assert d.get("ess_min") is not None
    assert d.get("rhat_max") is not None
    # mcse dict present
    assert isinstance(d.get("mcse"), dict)
