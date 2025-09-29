import numpy as np
import pytest

from core.uncertainty import bayesian_ci


def _fixture(n=64, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    a_true, b_true = 1.3, 0.4
    y = a_true + b_true * x + rng.normal(scale=sigma, size=n)
    X = np.column_stack([np.ones_like(x), x])
    th_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    th_hat = np.asarray(th_hat, float)

    def model(theta):
        theta = np.asarray(theta, float)
        return theta[0] + theta[1] * x

    return x, y, th_hat, model


def _extract_stat_vector(stats: dict, name: str):
    blk = stats[name]
    return np.array(
        [blk["est"], blk["sd"], blk["p2.5"], blk["p97.5"]],
        dtype=float,
    )


@pytest.mark.skipif(pytest.importorskip("emcee") is None, reason="emcee not available")
def test_bayesian_seed_is_deterministic():
    x, y, theta_hat, model = _fixture()

    common_kwargs = dict(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        fit_ctx={"alpha": 0.1, "bayes_diagnostics": False},
        n_walkers=16,
        n_burn=80,
        n_steps=160,
        thin=2,
        workers=None,
        return_band=False,
        param_names=["a", "b"],
    )

    r1 = bayesian_ci(seed=123, **common_kwargs)
    r2 = bayesian_ci(seed=123, **common_kwargs)

    v1_a = _extract_stat_vector(r1.stats, "a")
    v2_a = _extract_stat_vector(r2.stats, "a")
    v1_b = _extract_stat_vector(r1.stats, "b")
    v2_b = _extract_stat_vector(r2.stats, "b")

    np.testing.assert_allclose(v1_a, v2_a, rtol=0, atol=0)
    np.testing.assert_allclose(v1_b, v2_b, rtol=0, atol=0)


@pytest.mark.skipif(pytest.importorskip("emcee") is None, reason="emcee not available")
def test_bayesian_different_seeds_change_output_somewhat():
    x, y, theta_hat, model = _fixture()

    common_kwargs = dict(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        fit_ctx={"alpha": 0.1, "bayes_diagnostics": False},
        n_walkers=16,
        n_burn=80,
        n_steps=160,
        thin=2,
        workers=None,
        return_band=False,
        param_names=["a", "b"],
    )

    r1 = bayesian_ci(seed=777, **common_kwargs)
    r2 = bayesian_ci(seed=778, **common_kwargs)

    # it's possible medians match to high precision by chance; compare full vectors
    v1 = np.concatenate([
        _extract_stat_vector(r1.stats, "a"),
        _extract_stat_vector(r1.stats, "b"),
    ])
    v2 = np.concatenate([
        _extract_stat_vector(r2.stats, "a"),
        _extract_stat_vector(r2.stats, "b"),
    ])
    assert np.any(np.abs(v1 - v2) > 1e-8), "Different seeds should change posterior summaries at least slightly"
