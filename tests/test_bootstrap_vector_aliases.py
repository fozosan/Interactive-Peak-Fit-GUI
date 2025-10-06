import numpy as np

from core.uncertainty import bootstrap_ci


def _linear_fixture(n=48, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    a_true, b_true = 1.2, -0.3
    y = a_true + b_true * x + rng.normal(scale=sigma, size=n)
    X = np.column_stack([np.ones_like(x), x])
    th_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    th_hat = np.asarray(th_hat, float)

    def model(theta):
        theta = np.asarray(theta, float)
        return theta[0] + theta[1] * x

    r = y - model(th_hat)
    J = np.column_stack([-np.ones_like(x), -x])
    return x, y, th_hat, r, J, model


def test_bootstrap_vector_stats_have_underscore_aliases():
    x, y, theta_hat, r, J, model = _linear_fixture()

    def _refit(th0, locked_mask, bounds, x_in, y_in):
        return np.asarray(th0, float), True

    fit_ctx = {
        "alpha": 0.1,
        "refit": _refit,
        "strict_refit": True,
        "peaks": [object()],
    }

    res = bootstrap_ci(
        theta=theta_hat,
        residual=r,
        jacobian=J,
        predict_full=model,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        param_names=["a", "b"],
        n_boot=64,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
        workers=None,
    )

    for key, blk in res.stats.items():
        if not isinstance(blk, dict):
            continue
        assert "p2.5" in blk and "p97.5" in blk
        assert "p2_5" in blk and "p97_5" in blk
        assert blk["p2_5"] == blk["p2.5"]
        assert blk["p97_5"] == blk["p97.5"]
