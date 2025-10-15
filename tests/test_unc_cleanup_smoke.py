import importlib
import importlib.util
import importlib.machinery
import sys
import numpy as np


def test_helpers_exist_and_basic_behave():
    u = importlib.import_module("core.uncertainty")
    for name in ("_norm_solver_and_sharing",
                 "_build_residual_vector",
                 "_relabel_by_center",
                 "_validate_vector_length"):
        assert hasattr(u, name), f"missing helper {name}"
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([0.5, 2.5, 2.5, 5.0])
    resid = y - yhat
    r = u._build_residual_vector(residual=resid, y_all=y, y_hat=yhat, mode="raw", center=False)
    assert np.allclose(r, resid)
    ref = np.array([10, 1, 2, 0.3, 50, 2, 3, 0.7])
    new = np.array([50, 2, 3, 0.7, 10, 1, 2, 0.3])
    reord = u._relabel_by_center(new, ref, block=4)
    assert np.allclose(reord, ref)


def test_bayes_sharing_default_off(monkeypatch):
    u = importlib.import_module("core.uncertainty")

    class DummySampler:
        def __init__(self, nwalkers, dim, log_prob, pool=None, random_state=None):
            self.nwalkers = nwalkers
            self.dim = dim
            self.log_prob = log_prob
            self.pool = pool
            self.acceptance_fraction = np.zeros(nwalkers)

        def run_mcmc(self, state, *a, **k):
            return state, None, None

        @property
        def chain(self):
            return np.zeros((2, 2, 2))

        @property
        def blobs(self):
            return None

        def get_chain(self, *a, **k):
            return np.zeros((1, self.nwalkers, self.dim))

        def get_blobs(self, *a, **k):
            return None

    dummy = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("emcee", None))
    dummy.EnsembleSampler = DummySampler
    monkeypatch.setitem(sys.modules, "emcee", dummy)

    theta = np.array([1, 2, 3, 0.5, 10, 4, 5, 0.6], dtype=float)
    x = np.linspace(0.0, 1.0, theta.size)
    y = np.ones_like(x)
    res = u.bayesian_ci(
        theta,
        fit_ctx={"alpha": 0.5},
        x_all=x,
        y_all=y,
        predict_full=lambda th: np.ones_like(x),
        return_band=False,
    )
    assert res is not None


def test_bootstrap_passes_bounds_and_locks(monkeypatch):
    u = importlib.import_module("core.uncertainty")

    captured = {}
    fit_api = importlib.import_module("core.fit_api")

    def _rfc(
        x,
        y,
        cfg=None,
        config=None,
        solver=None,
        peaks_in=None,
        peaks=None,
        baseline=None,
        mode=None,
        fit_mask=None,
        rng_seed=None,
        verbose=None,
        quick_and_dirty=None,
        locked_mask=None,
        bounds=None,
        theta_init=None,
        **extra,
    ):
        kwargs = {
            "x": x,
            "y": y,
            "cfg": cfg,
            "config": config,
            "solver": solver,
            "peaks_in": peaks_in,
            "peaks": peaks,
            "baseline": baseline,
            "mode": mode,
            "fit_mask": fit_mask,
            "rng_seed": rng_seed,
            "verbose": verbose,
            "quick_and_dirty": quick_and_dirty,
            "locked_mask": locked_mask,
            "bounds": bounds,
            "theta_init": theta_init,
        }
        kwargs.update(extra)
        captured.setdefault("kwargs", []).append(kwargs)
        theta_init_arr = np.asarray(theta_init if theta_init is not None else np.zeros(4), float)
        return {"theta": theta_init_arr}

    monkeypatch.setattr(fit_api, "run_fit_consistent", _rfc, raising=True)

    theta = np.array([10.0, 1.0, 2.0, 0.5])
    x = np.linspace(0, 1, 16)
    y = np.ones_like(x)
    resid = y - y
    J = np.eye(x.size, theta.size)

    out = u.bootstrap_ci(
        theta,
        jacobian=J,
        residual=resid,
        x_all=x,
        y_all=y,
        predict_full=lambda th: np.ones_like(x),
        fit_ctx={
            "bounds": (np.full_like(theta, -np.inf), np.full_like(theta, np.inf)),
            "locked_mask": np.array([False, True, False, False]),
            "unc_workers": 0,
            "n_boot": 2,
            "allow_linear_fallback": False,
        },
        n_boot=2,
        workers=0,
        return_band=False,
    )
    assert out is not None
    calls = captured.get("kwargs", [])
    assert calls, "run_fit_consistent was not invoked"
    assert any(
        "bounds" in kw and "locked_mask" in kw
        for kw in calls
    ), f"expected bounds/locked_mask in calls, saw: {calls}"
