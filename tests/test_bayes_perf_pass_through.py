import os
import numpy as np
import pytest

from uncertainty import bayes as ub
from core.peaks import Peak

pytest.importorskip("emcee")


def test_bayes_blas_env_clamp_pass_through(monkeypatch):
    x = np.linspace(0.0, 1.0, 32)
    a_true, b_true = 0.9, 0.05
    y = a_true * x + b_true
    theta0 = np.array([a_true, b_true, 0.1, 0.2], float)
    template_peak = Peak(theta0[0], theta0[1], theta0[2], theta0[3], False, False)

    captured = {"mkl": None, "openblas": None, "omp": None}

    from core.residuals import build_residual as real_builder

    def _build_residual(x_, y_, peaks_, mode, baseline, like, opts):
        inner = real_builder(x_, y_, peaks_, mode, baseline, "linear", None)

        def capturing_residual(theta):
            if captured["mkl"] is None:
                captured["mkl"] = os.environ.get("MKL_NUM_THREADS")
                captured["openblas"] = os.environ.get("OPENBLAS_NUM_THREADS")
                captured["omp"] = os.environ.get("OMP_NUM_THREADS")
            return inner(theta)

        return capturing_residual

    monkeypatch.setattr(ub, "build_residual", _build_residual)

    priors = {"lb": -np.inf, "ub": np.inf, "sigma": 1.0}
    init = {
        "x": x,
        "y": y,
        "peaks": [template_peak],
        "mode": "add",
        "baseline": None,
        "theta": theta0,
    }

    sampler_cfg = {
        "nwalkers": 8,
        "nsteps": 20,
        "seed": 42,
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
    }
    ub.bayesian(priors, "gaussian", init, sampler_cfg, None)
    assert captured["mkl"] == "1"
    assert captured["openblas"] == "1"
    assert captured["omp"] == "1"

    for k in captured:
        captured[k] = None

    sampler_cfg2 = {
        "nwalkers": 8,
        "nsteps": 20,
        "seed": 42,
        "perf_parallel_strategy": "inner",
        "perf_blas_threads": 3,
    }
    ub.bayesian(priors, "gaussian", init, sampler_cfg2, None)
    assert captured["mkl"] == "3"
    assert captured["openblas"] == "3"
    assert captured["omp"] == "3"
