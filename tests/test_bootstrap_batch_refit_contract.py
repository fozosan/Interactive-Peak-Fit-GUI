import numpy as np
import pytest
from core.uncertainty import bootstrap_ci


def test_bootstrap_batch_refit_requires_ok_flag():
    x = np.linspace(0, 1, 32)
    theta = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.sin(2*np.pi*x)
    resid = y - y.mean()
    J = np.ones((x.size, theta.size))

    calls = {"n": 0}

    def ok_refit(th0, locked_mask, bounds, x_in, y_in):
        calls["n"] += 1
        return np.asarray(th0, float), True

    fit_ctx_ok = {
        "refit": ok_refit,
        "peaks": [object()],
        "allow_linear_fallback": False,
        "strict_refit": True,
    }
    out = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx_ok,
        n_boot=8,
        seed=0,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    assert out.diagnostics["n_success"] == 8
    assert calls["n"] >= 1

    def bare_refit(th0, locked_mask, bounds, x_in, y_in):
        return np.asarray(th0, float)  # no ok flag

    with pytest.raises(RuntimeError) as excinfo:
        bootstrap_ci(
            theta=theta,
            residual=resid,
            jacobian=J,
            predict_full=lambda th: y,
            x_all=x,
            y_all=y,
            fit_ctx={
                "refit": bare_refit,
                "peaks": [object()],
                "allow_linear_fallback": False,
                "strict_refit": True,
            },
            n_boot=8,
            seed=0,
            workers=None,
            alpha=0.1,
            center_residuals=True,
            return_band=False,
        )
    assert "success=0" in str(excinfo.value)
