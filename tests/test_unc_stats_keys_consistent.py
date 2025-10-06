import numpy as np
import pytest

from core.uncertainty import bootstrap_ci, bayesian_ci


def _linear_fixture(n=60, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    a_true, b_true = 1.5, 0.5
    y = a_true + b_true * x + rng.normal(scale=sigma, size=n)

    # LS estimate for theta_hat = [a, b]
    X = np.column_stack([np.ones_like(x), x])
    th_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    th_hat = np.asarray(th_hat, float)

    def model(theta):
        theta = np.asarray(theta, float)
        return theta[0] + theta[1] * x

    r = y - model(th_hat)
    J = np.column_stack([-np.ones_like(x), -x])  # residual Jacobian wrt [a,b]

    return x, y, th_hat, r, J, model


def _assert_param_block_has_consistent_keys(block: dict):
    # Ensure dotted quantile keys exist and, if underscored forms are present, they match
    assert {"est", "sd", "p2.5", "p97.5"} <= set(block.keys())
    if "p2_5" in block:
        assert pytest.approx(block["p2.5"]) == block["p2_5"]
    if "p97_5" in block:
        assert pytest.approx(block["p97.5"]) == block["p97_5"]


def test_stats_keys_consistent_between_bootstrap_and_bayesian():
    x, y, theta_hat, r, J, model = _linear_fixture()

    # --- Bootstrap (linearized path; band off for speed) ---
    br = bootstrap_ci(
        theta=theta_hat,
        residual=r,
        jacobian=J,
        predict_full=model,
        x_all=x,
        y_all=y,
        fit_ctx={"alpha": 0.1},
        param_names=["a", "b"],
        n_boot=64,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
        workers=None,
    )
    bstats = br.stats

    def _fetch_block(stats_dict, label, fallback_index):
        if label in stats_dict:
            return stats_dict[label]
        fallback_key = f"p{fallback_index}"
        assert fallback_key in stats_dict
        return stats_dict[fallback_key]

    _assert_param_block_has_consistent_keys(_fetch_block(bstats, "a", 0))
    _assert_param_block_has_consistent_keys(_fetch_block(bstats, "b", 1))

    # --- Bayesian (small run; band/diagnostics off) ---
    try:
        import emcee  # noqa: F401
    except Exception:
        pytest.skip("emcee not available")

    res = bayesian_ci(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        fit_ctx={"alpha": 0.1, "bayes_diagnostics": False},
        param_names=["a", "b"],
        n_walkers=16,
        n_burn=100,
        n_steps=150,
        thin=5,
        seed=123,
        workers=None,
        return_band=False,
    )
    stats = res.stats
    _assert_param_block_has_consistent_keys(_fetch_block(stats, "a", 0))
    _assert_param_block_has_consistent_keys(_fetch_block(stats, "b", 1))
