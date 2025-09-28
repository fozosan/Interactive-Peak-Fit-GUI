import numpy as np
import pytest

from core.uncertainty import bootstrap_ci


def test_nan_refits_are_not_counted(monkeypatch):
    # Minimal, deterministic inputs
    n = 8
    theta = np.array([1.0, 2.0, 3.0, 0.5])
    residual = np.zeros(n)
    J = np.ones((n, theta.size))

    # Refits return NaNs but claim success -> should be treated as failure
    def bad_refit(theta_init, x, y):
        return (np.full_like(theta_init, np.nan), True)

    fit_ctx = {
        "x_all": np.arange(n, dtype=float),
        "y_all": np.zeros(n),
        "residual_fn": lambda th: np.zeros(n),
        "predict_full": lambda th: np.zeros(n),
        "strict_refit": True,
        "refit": bad_refit,
    }

    with pytest.raises(RuntimeError, match="Insufficient successful bootstrap refits"):
        bootstrap_ci(
            theta=theta,
            residual=residual,
            jacobian=J,
            predict_full=lambda th: np.zeros(n),
            x_all=np.arange(n, dtype=float),
            y_all=np.zeros(n),
            fit_ctx=fit_ctx,
            n_boot=3,
            seed=123,
            workers=None,
            alpha=0.05,
            center_residuals=True,
            return_band=False,
        )
