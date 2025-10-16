import importlib
import importlib.util
import importlib.machinery
import sys
import numpy as np
import types

from core.peaks import Peak


def test_helpers_exist_and_basic_behave():
    u = importlib.import_module("core.uncertainty")
    for name in (
        "_norm_solver_and_sharing",
        "_build_residual_vector",
        "_relabel_by_center",
        "_validate_vector_length",
    ):
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
            self.random_state = random_state
            self.acceptance_fraction = np.full(nwalkers, 0.5, dtype=float)
            self._calls = []

        def run_mcmc(self, state, step, **kwargs):
            self._calls.append(np.asarray(state, float))
            return state, None, None

        def get_chain(self, discard=0, thin=1, flat=False):
            steps = max(len(self._calls), discard + 2)
            chain = np.zeros((steps, self.nwalkers, self.dim), float)
            return chain[discard::thin]

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

    # Stub fit_api.run_fit_consistent to capture kwargs
    captured = {}
    theta = np.array([10.0, 1.0, 2.0, 0.5])

    def _rfc(
        x,
        y,
        peaks_in=None,
        cfg=None,
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
            "peaks_in": peaks_in,
            "cfg": cfg,
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
        captured["kwargs"] = kwargs
        theta_init = np.asarray(theta_init if theta_init is not None else np.zeros_like(theta), float)
        return {"theta": theta_init}

    fit_api_module = importlib.import_module("core.fit_api")
    monkeypatch.setattr(fit_api_module, "run_fit_consistent", _rfc, raising=True)

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
        },
        n_boot=2,
        workers=0,
        return_band=False,
    )
    assert out is not None
    kw = captured.get("kwargs", {})
    assert "bounds" in kw and "locked_mask" in kw


def test_fit_api_no_solver_fallbacks(monkeypatch):
    import core.fit_api as fit_api

    called = {"once": 0, "fb": 0}

    def _once(x_fit, y_solver, peaks_start, mode, base_solver, opts):
        called["once"] += 1
        return types.SimpleNamespace(
            theta=np.asarray(theta0, float),
            peaks_out=peaks_start,
            success=True,
            solver=opts.get("solver", "modern_trf"),
        )

    def _fb(x_fit, y_solver, peaks_start, mode, base_solver, opts):
        called["fb"] += 1
        return types.SimpleNamespace(
            theta=np.asarray(theta0, float),
            peaks_out=peaks_start,
            success=True,
            solver=opts.get("solver", "modern_trf"),
        )

    dummy_orch = types.SimpleNamespace(step_once=_once, run_fit_with_fallbacks=_fb)
    monkeypatch.setattr(fit_api, "orchestrator", dummy_orch, raising=True)

    x = np.linspace(0, 1, 8)
    y = np.ones_like(x)
    peaks_in = [Peak(center=0.4, height=1.0, fwhm=0.2, eta=0.5)]
    theta0 = np.array([p.center for p in peaks_in] + [p.height for p in peaks_in] + [p.fwhm for p in peaks_in] + [p.eta for p in peaks_in])
    res = fit_api.run_fit_consistent(
        x=x,
        y=y,
        peaks_in=peaks_in,
        cfg={"solver": "modern_trf", "no_solver_fallbacks": True},
        baseline=None,
        mode="add",
        fit_mask=np.ones_like(x, bool),
    )
    assert called["once"] == 1 and called["fb"] == 0
    assert res.get("theta") is not None
