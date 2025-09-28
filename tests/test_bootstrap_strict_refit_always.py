import numpy as np
from types import SimpleNamespace

from core.uncertainty import bootstrap_ci


def test_bootstrap_always_refits_even_without_jitter():
    # Minimal setup: one "peak" with 4 params; free mask non-empty
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x)
    theta = np.array([0.3, 1.0, 0.2, 0.5])  # [c, h, w, eta]
    resid = y - y.mean()
    # Jacobian shape (N, P) with non-zero columns on free params
    J = np.ones((x.size, theta.size))

    called = {"count": 0}

    def fake_refit(theta_init, locked_mask, bounds, x_in, y_in):
        called["count"] += 1
        # Return theta_init, ok=True to simulate a successful refit
        return np.asarray(theta_init, float), True

    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "baseline": None,
        "mode": "add",
        "refit": fake_refit,
        "peaks": [
            SimpleNamespace(
                center=float(theta[0]),
                height=float(theta[1]),
                fwhm=float(theta[2]),
                eta=float(theta[3]),
                lock_center=False,
                lock_width=False,
            )
        ],
        "bootstrap_jitter": 0.0,    # jitter = 0
        "strict_refit": True,        # enforced by callers in production; explicit here for safety
        "allow_linear_fallback": True,
    }

    res = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=8,
        seed=0,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    assert res.diagnostics.get("bootstrap_mode") == "refit"
    assert called["count"] >= 1
