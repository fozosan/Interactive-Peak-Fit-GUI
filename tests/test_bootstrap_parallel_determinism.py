import numpy as np

from core.uncertainty import bootstrap_ci


def test_bootstrap_parallel_is_deterministic_with_seed():
    x = np.linspace(0, 1, 64)
    theta = np.array([0.3, 1.0, 0.2, 0.5])
    rng = np.random.default_rng(0)
    y = np.sin(2 * np.pi * x) + 0.01 * rng.normal(size=x.size)
    resid = y - y.mean()
    J = np.ones((x.size, theta.size))

    def refit(th0, locked_mask, bounds, x_in, y_in):
        # Deterministic: return jittered init as the "fit" with ok=True
        return np.asarray(th0, float), True

    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "baseline": None,
        "mode": "add",
        "refit": refit,
        "strict_refit": True,
        "bootstrap_jitter": 0.05,
        "peaks": [object()],
    }

    r_serial = bootstrap_ci(
        theta,
        resid,
        J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=64,
        seed=42,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )

    r_parallel = bootstrap_ci(
        theta,
        resid,
        J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=64,
        seed=42,
        workers=4,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )

    assert r_serial.diagnostics["n_success"] == 64
    assert r_parallel.diagnostics["n_success"] == 64
    np.testing.assert_allclose(
        r_serial.stats["p0"]["est"],
        r_parallel.stats["p0"]["est"],
    )
