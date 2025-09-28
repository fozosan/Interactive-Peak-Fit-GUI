import numpy as np
import pytest

from core import uncertainty as U


def _fake_predict(th, x):
    return np.zeros_like(x)


def _fake_resid(th, x):
    return np.zeros_like(x)


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
