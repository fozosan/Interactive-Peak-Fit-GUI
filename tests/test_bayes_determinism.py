import numpy as np
import pytest

from infra import performance
from uncertainty import bayes as ub
from core.peaks import Peak

pytest.importorskip("emcee")


def _residual_builder_factory(x, y):
    def _residual(theta):
        a = float(theta[0])
        b = float(theta[1])
        return a * x + b - y

    return _residual


def test_bayes_determinism_seed_all_true(monkeypatch):
    x = np.linspace(0.0, 1.0, 64)
    a_true, b_true = 1.2, -0.3
    y = a_true * x + b_true

    theta0 = np.array([a_true, b_true, 0.1, 0.2], float)
    template_peak = Peak(theta0[0], theta0[1], theta0[2], theta0[3], False, False)

    def _build_residual(x_, y_, peaks_, mode, baseline, like, opts):
        return _residual_builder_factory(x_, y_)

    monkeypatch.setattr(ub, "build_residual", _build_residual)

    priors = {"lb": -np.inf, "ub": np.inf, "sigma": 1.0}
    init = {
        "x": x,
        "y": y,
        "peaks": [template_peak],
        "mode": "add",
        "baseline": None,
        "theta": theta0,
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
    }
    sampler_cfg = {
        "nwalkers": 8,
        "nsteps": 200,
        "seed": None,
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
    }

    performance.apply_global_seed(777, True)
    out1 = ub.bayesian(priors, "gaussian", init, sampler_cfg, None)
    performance.apply_global_seed(777, True)
    out2 = ub.bayesian(priors, "gaussian", init, sampler_cfg, None)

    t1 = np.asarray(out1["params"]["theta"], float)
    t2 = np.asarray(out2["params"]["theta"], float)
    c1 = np.asarray(out1["params"]["cov"], float)
    c2 = np.asarray(out2["params"]["cov"], float)

    assert np.allclose(t1, t2)
    assert np.allclose(c1, c2)
