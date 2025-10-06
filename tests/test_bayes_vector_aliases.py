import numpy as np
import pytest

from core.uncertainty import bayesian_ci

pytest.importorskip("emcee")


def test_bayesian_vector_stats_have_underscore_aliases():
    # Simple linear model: y = a*x + b
    x = np.linspace(0.0, 1.0, 32)
    a_true, b_true = 1.2, -0.3
    y = a_true * x + b_true
    theta_hat = np.array([a_true, b_true, 0.1, 0.2], float)

    def predict_full(theta):
        a = float(theta[0])
        b = float(theta[1])
        return a * x + b

    r = bayesian_ci(
        theta_hat,
        predict_full=predict_full,
        x_all=x,
        y_all=y,
        fit_ctx={
            "bayes_band_enabled": False,
            "bayes_sampler_cfg": {"nwalkers": 8, "nsteps": 50, "seed": 0},
            "perf_parallel_strategy": "outer",
            "perf_blas_threads": 0,
        },
        return_band=False,
    )

    # Vector blocks should include both dotted and underscored keys, and match numerically
    for key in ("center", "height", "fwhm", "eta"):
        blk = r.stats[key]
        assert {"p2.5", "p97.5", "p2_5", "p97_5"} <= set(blk.keys())
        assert blk["p2_5"] == blk["p2.5"]
        assert blk["p97_5"] == blk["p97.5"]
        # also basic shape checks (lists of equal length)
        assert len(blk["p2_5"]) == len(blk["p97_5"]) == len(blk["est"]) == len(blk["sd"])
