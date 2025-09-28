import numpy as np
import pytest

from core.uncertainty import bootstrap_ci


def test_bootstrap_raises_when_refits_fail():
    x = np.linspace(0, 1, 32)
    theta = np.array([0.4, 0.8, 0.25, 0.5])
    y = np.sin(2 * np.pi * (x - theta[0]))
    resid = y - y
    J = np.ones((x.size, theta.size))

    def always_fail(theta_init, locked_mask, bounds, x_in, y_in):
        return np.array(theta_init, float), False

    with pytest.raises(RuntimeError) as exc:
        bootstrap_ci(
            theta=theta,
            residual=resid,
            jacobian=J,
            predict_full=lambda th: y,
            x_all=x,
            y_all=y,
            fit_ctx={
                "refit": always_fail,
                "bootstrap_jitter": 0.05,
                "strict_refit": True,
                "peaks": [object()],
            },
            n_boot=8,
            seed=123,
            workers=None,
            alpha=0.1,
            center_residuals=True,
            return_band=False,
        )
    assert "Insufficient successful" in str(exc.value)
