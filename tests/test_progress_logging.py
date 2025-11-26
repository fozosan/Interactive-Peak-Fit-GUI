import numpy as np
import pytest

from core import uncertainty as U
from tests.conftest import bayes_knobs, ensure_unc_common


class _DummySampler:
    def __init__(self, nwalkers, dim, log_prob_fn, pool=None, random_state=None):
        self.acceptance_fraction = np.full(nwalkers, 0.3)
        self._state = np.zeros((nwalkers, dim))
        self._chain = np.zeros((5000, nwalkers, dim), dtype=float)
        self._pos = 0

    def run_mcmc(self, state, step, progress=False, skip_initial_state_check=True):
        start = self._pos
        end = min(self._chain.shape[0], start + int(step))
        if end > start:
            self._chain[start:end] = 0.5
        self._pos = end
        return state, None, None

    def get_chain(self, discard=0, thin=1, flat=False):
        arr = self._chain
        if discard:
            arr = arr[int(discard) :]
        if thin and thin > 1:
            arr = arr[:: int(thin)]
        return arr if not flat else arr.reshape(-1, arr.shape[-1])


class _AbortEvent:
    def is_set(self):
        return False


def test_bayesian_progress_logging_accepts_total_steps(monkeypatch):
    emcee = pytest.importorskip("emcee")
    monkeypatch.setattr(emcee, "EnsembleSampler", _DummySampler)

    x = np.linspace(0, 1, 16)
    y = np.zeros_like(x)
    theta = np.array([0.2, 1.5, 0.1, 0.4])

    logs: list[str] = []
    fit_ctx = ensure_unc_common(
        {
            **bayes_knobs(walkers=4, burn=2500, steps=2500, thin=1),
            "progress_cb": logs.append,
            "abort_event": _AbortEvent(),
        }
    )
    fit_ctx.setdefault("bayes_prior_sigma", "half_cauchy")
    fit_ctx.setdefault("alpha", 0.05)

    U.bayesian_ci(
        theta_hat=theta,
        model=lambda th: np.zeros_like(x),
        predict_full=lambda th: np.zeros_like(x),
        x_all=x,
        y_all=y,
        residual_fn=lambda th: np.zeros_like(x),
        seed=123,
        fit_ctx=fit_ctx,
        n_walkers=4,
        n_burn=2500,
        n_steps=2500,
        thin=1,
        workers=0,
        return_band=False,
    )

    log_text = "\n".join(logs)
    assert "Bayesian MCMC: 100/5000" in log_text or "Bayesian MCMC: 100/total_steps" in log_text

