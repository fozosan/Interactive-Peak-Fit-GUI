import numpy as np
import pytest

from infra import performance
from core.uncertainty import bayesian_ci

pytest.importorskip("emcee")


def test_bayes_ci_determinism_seed_all_true():
    # Simple linear toy model: y = a*x + b
    x = np.linspace(0.0, 1.0, 64)
    a_true, b_true = 1.2, -0.3
    y = a_true * x + b_true
    theta_hat = np.array([a_true, b_true, 0.1, 0.2], float)

    def predict_full(theta):
        a = float(theta[0])
        b = float(theta[1])
        return a * x + b

    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "bayes_sampler_cfg": {"nwalkers": 8, "nsteps": 200, "seed": None},
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
        "bayes_band_enabled": False,
    }

    performance.apply_global_seed(777, True)
    r1 = bayesian_ci(
        theta_hat,
        predict_full=predict_full,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        return_band=False,
    )

    performance.apply_global_seed(777, True)
    r2 = bayesian_ci(
        theta_hat,
        predict_full=predict_full,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        return_band=False,
    )

    assert r1.stats == r2.stats
    assert r1.diagnostics.get("seed") == r2.diagnostics.get("seed") == 777
