import numpy as np

from core.uncertainty import bootstrap_ci


def test_jitter_changes_theta_init_on_free_params():
    n = 12
    theta = np.array([5.0, 2.0, 1.0, 0.5])
    residual = np.zeros(n)
    J = np.ones((n, theta.size))
    x = np.arange(n, dtype=float)
    y = np.zeros(n)

    captured = []

    def capturing_refit(theta_init, x_in, y_in):
        captured.append(np.asarray(theta_init, float).copy())
        # echo back θ so we count success
        return (np.asarray(theta_init, float), True)

    # With jitter=0 the starts are identical
    fit_ctx_zero = dict(
        x_all=x, y_all=y,
        residual_fn=lambda th: np.zeros_like(x),
        predict_full=lambda th: np.zeros_like(x),
        strict_refit=True,
        refit=capturing_refit,
    )
    captured.clear()
    bootstrap_ci(theta, residual, J, predict_full=lambda th: np.zeros_like(x),
                 x_all=x, y_all=y, fit_ctx=fit_ctx_zero,
                 n_boot=3, seed=123, workers=None,
                 alpha=0.05, center_residuals=True, return_band=False)
    # All captured starts equal to theta
    assert all(np.allclose(t, theta) for t in captured)

    # With jitter > 0, at least some starts should differ on free params
    fit_ctx_jit = dict(fit_ctx_zero)
    captured.clear()
    bootstrap_ci(theta, residual, J, predict_full=lambda th: np.zeros_like(x),
                 x_all=x, y_all=y, fit_ctx=fit_ctx_jit,
                 n_boot=6, seed=123, workers=None,
                 alpha=0.05, center_residuals=True, return_band=False,
                 )
    # Default jitter in engine is 0 unless provided; emulate GUI by setting it:
    # Re-run with explicit jitter
    captured.clear()
    bootstrap_ci(theta, residual, J, predict_full=lambda th: np.zeros_like(x),
                 x_all=x, y_all=y,
                 fit_ctx=fit_ctx_jit | {"bootstrap_jitter": 0.25},
                 n_boot=6, seed=123, workers=None,
                 alpha=0.05, center_residuals=True, return_band=False)
    assert any(not np.allclose(t, theta) for t in captured)
