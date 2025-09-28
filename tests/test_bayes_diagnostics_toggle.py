import numpy as np

from core import uncertainty as U


class DummySampler:
    def __init__(self, n_walkers, dim, log_prob, pool=None):
        self.n_walkers = n_walkers
        self.dim = dim
        self.acceptance_fraction = np.full(n_walkers, 0.25)
        # create deterministic chain (steps, walkers, dim)
        steps = 40
        base = np.linspace(-1.0, 1.0, steps * n_walkers * dim)
        self._chain = base.reshape(steps, n_walkers, dim)

    def run_mcmc(self, state, step, progress=False, skip_initial_state_check=True):
        return state, None, None

    def get_chain(self, discard=0, thin=1, flat=False):
        if flat:
            return self._chain.reshape(-1, self.dim)
        return self._chain


def _setup_sampler(monkeypatch):
    import emcee

    monkeypatch.setattr(emcee, "EnsembleSampler", DummySampler)


def _basic_args():
    x = np.linspace(0, 1, 30)
    y = np.zeros_like(x)
    theta = np.array([0.1, 1.0, 0.5, 0.4])
    model = lambda th: np.zeros_like(x)
    residual_fn = lambda th: np.zeros_like(x)
    ctx = {"x_all": x, "y_all": y, "baseline": np.zeros_like(x), "mode": "add"}
    return theta, x, y, model, residual_fn, ctx


def test_bayes_diag_toggle_off(monkeypatch):
    _setup_sampler(monkeypatch)
    monkeypatch.setattr("core.mcmc_utils.ess_autocorr", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ESS called")))
    monkeypatch.setattr("core.mcmc_utils.rhat_split", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Rhat called")))

    theta, x, y, model, residual_fn, ctx = _basic_args()
    res = U.bayesian_ci(
        theta_hat=theta,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=residual_fn,
        fit_ctx={**ctx, "bayes_diagnostics": False},
        n_walkers=3,
        n_burn=10,
        n_steps=30,
        thin=2,
        seed=42,
        workers=0,
        return_band=False,
    )
    diag = res.diagnostics or {}
    assert diag.get("diagnostics_enabled") is False
    assert np.isnan(diag.get("ess_min"))
    assert np.isnan(diag.get("rhat_max"))
    assert np.isnan(diag.get("mcse_mean"))


def test_bayes_diag_toggle_on(monkeypatch):
    _setup_sampler(monkeypatch)
    monkeypatch.setattr("core.mcmc_utils.ess_autocorr", lambda post: 60.0)
    monkeypatch.setattr("core.mcmc_utils.rhat_split", lambda post: 1.05)

    theta, x, y, model, residual_fn, ctx = _basic_args()
    res = U.bayesian_ci(
        theta_hat=theta,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=residual_fn,
        fit_ctx={**ctx, "bayes_diagnostics": True},
        n_walkers=3,
        n_burn=10,
        n_steps=30,
        thin=2,
        seed=99,
        workers=0,
        return_band=False,
    )
    diag = res.diagnostics or {}
    assert diag.get("diagnostics_enabled") is True
    assert np.isfinite(diag.get("ess_min"))
    assert np.isfinite(diag.get("rhat_max"))
    assert "mcse_mean" in diag
