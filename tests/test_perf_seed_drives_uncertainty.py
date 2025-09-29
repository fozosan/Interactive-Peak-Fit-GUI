import numpy as np
import pytest

from core import uncertainty as U


def _fake_predict(th, x):
    return np.zeros_like(x)


def _fake_resid(th, x):
    return np.zeros_like(x)


class _DummySamplerBand:
    def __init__(self, nwalkers, dim, log_prob_fn, pool=None):
        self._log_prob = log_prob_fn
        self.pool = pool
        self.nwalkers = nwalkers
        self.dim = dim
        self.acceptance_fraction = np.full(nwalkers, 0.3)
        rng = np.random.RandomState(1234)
        self._chain = rng.normal(size=(128, nwalkers, dim))

    def run_mcmc(self, state, step, progress=False, skip_initial_state_check=True):
        return state, None, None

    def get_chain(self, discard=0, thin=1, flat=False):
        arr = self._chain
        if discard:
            arr = arr[discard:]
        if thin and thin > 1:
            arr = arr[::thin]
        return arr if not flat else arr.reshape(-1, arr.shape[-1])


def test_bootstrap_uses_global_seed(monkeypatch):
    x = np.linspace(0, 1, 64)
    y = np.zeros_like(x)
    theta = np.array([1.0, 2.0, 3.0, 0.5])

    def refit(theta_init, x, y, **kw):
        return {"theta": np.asarray(theta_init, float), "fit_ok": True}

    monkeypatch.setattr(U, "refit", refit, raising=False)

    out1 = U.bootstrap_ci(
        theta=theta,
        residual=y,
        jacobian=np.zeros((x.size, theta.size)),
        predict_full=lambda th: _fake_predict(th, x),
        x_all=x,
        y_all=y,
        seed=123,
        n_boot=8,
        fit_ctx={},
        workers=None,
        return_band=False,
    )
    out2 = U.bootstrap_ci(
        theta=theta,
        residual=y,
        jacobian=np.zeros((x.size, theta.size)),
        predict_full=lambda th: _fake_predict(th, x),
        x_all=x,
        y_all=y,
        seed=123,
        n_boot=8,
        fit_ctx={},
        workers=None,
        return_band=False,
    )
    assert out1.stats == out2.stats


def test_bayesian_respects_seed(monkeypatch):
    emcee = pytest.importorskip("emcee")

    class DummySampler:
        def __init__(self, *a, **k):
            self.acceptance_fraction = np.array([0.25, 0.25])

        def run_mcmc(self, state, step, **kw):
            return state, None, None

        def get_chain(self, discard=0, thin=1, flat=False):
            rng = np.random.RandomState(1234)
            arr = rng.randn(40, 2, 3)
            if discard:
                arr = arr[discard:]
            if thin and thin > 1:
                arr = arr[::thin]
            return arr if not flat else arr.reshape(-1, 3)

    monkeypatch.setattr(emcee, "EnsembleSampler", DummySampler)

    x = np.linspace(0, 1, 32)
    y = np.zeros_like(x)
    theta = np.array([0.1, 1.0, 0.5, 0.4])

    r1 = U.bayesian_ci(
        theta_hat=theta,
        model=lambda th: _fake_predict(th, x),
        predict_full=lambda th: _fake_predict(th, x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: _fake_resid(th, x),
        seed=99,
        fit_ctx={},
        workers=0,
        return_band=False,
    )
    r2 = U.bayesian_ci(
        theta_hat=theta,
        model=lambda th: _fake_predict(th, x),
        predict_full=lambda th: _fake_predict(th, x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: _fake_resid(th, x),
        seed=99,
        fit_ctx={},
        workers=0,
        return_band=False,
    )
    assert r1.stats == r2.stats


def test_bayesian_band_produces_subset_when_forced(monkeypatch):
    emcee = pytest.importorskip("emcee")
    monkeypatch.setattr(emcee, "EnsembleSampler", _DummySamplerBand)

    x = np.linspace(0, 1, 64)
    y = np.zeros_like(x)
    theta = np.array([0.2, 1.5, 0.1, -0.05])
    logs: list[str] = []

    def predict(th):
        return th[0] + th[1] * x

    fit_ctx = {
        "alpha": 0.1,
        "bayes_band_enabled": True,
        "bayes_band_force": True,
        "progress_cb": logs.append,
    }

    res = U.bayesian_ci(
        theta_hat=theta,
        model=predict,
        predict_full=predict,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: _fake_resid(th, x),
        seed=123,
        fit_ctx=fit_ctx,
        n_walkers=4,
        n_burn=1,
        n_steps=64,
        thin=2,
        workers=0,
        return_band=True,
    )

    assert res.band is not None
    xb, lo, hi = res.band
    assert xb.shape == x.shape
    assert lo.shape == x.shape
    assert hi.shape == x.shape
    assert res.diagnostics["band_source"] == "posterior_predictive_subset"
    assert res.diagnostics["band_gated"] is False
    assert res.diagnostics.get("band_forced") is True
    assert res.diagnostics["band_draws_used"] <= 512
    assert any("Bayesian band" in msg for msg in logs)


def test_bayesian_band_gated_without_force(monkeypatch):
    emcee = pytest.importorskip("emcee")
    monkeypatch.setattr(emcee, "EnsembleSampler", _DummySamplerBand)

    x = np.linspace(0, 1, 32)
    y = np.zeros_like(x)
    theta = np.array([0.2, 1.5, 0.1, -0.05])

    def predict(th):
        return th[0] + th[1] * x

    fit_ctx = {
        "bayes_band_enabled": True,
        "bayes_diagnostics": True,
        "bayes_diag_ess_min": 1e9,
        "bayes_diag_rhat_max": 0.9,
        "bayes_diag_mcse_mean": 1e-6,
    }

    res = U.bayesian_ci(
        theta_hat=theta,
        model=predict,
        predict_full=predict,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: _fake_resid(th, x),
        seed=321,
        fit_ctx=fit_ctx,
        n_walkers=4,
        n_burn=1,
        n_steps=64,
        thin=2,
        workers=0,
        return_band=True,
    )

    assert res.band is None
    assert res.diagnostics["band_gated"] is True
    assert res.diagnostics.get("band_forced") is False
    assert res.diagnostics["band_source"] is None
    assert res.diagnostics["band_draws_used"] > 0

    fit_ctx_forced = dict(fit_ctx, bayes_band_force=True)
    res_forced = U.bayesian_ci(
        theta_hat=theta,
        model=predict,
        predict_full=predict,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: _fake_resid(th, x),
        seed=321,
        fit_ctx=fit_ctx_forced,
        n_walkers=4,
        n_burn=1,
        n_steps=64,
        thin=2,
        workers=0,
        return_band=True,
    )

    assert res_forced.band is not None
    assert res_forced.diagnostics["band_gated"] is False
    assert res_forced.diagnostics.get("band_forced") is True
