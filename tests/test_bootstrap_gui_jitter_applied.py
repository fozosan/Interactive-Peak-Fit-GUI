import numpy as np

from core.uncertainty import bootstrap_ci


def test_gui_like_single_file_jitter_is_applied():
    x = np.linspace(0, 1, 64)
    theta = np.array([0.3, 1.0, 0.2, 0.5])
    y = np.sin(2 * np.pi * (x - theta[0]))
    resid = y - (y * 0 + y)
    J = np.ones((x.size, theta.size))

    calls = {"n": 0}

    def accept_jitter_refit(theta_init, locked_mask, bounds, x_in, y_in):
        calls["n"] += 1
        return np.array(theta_init, float), True

    out = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx={
            "refit": accept_jitter_refit,
            "bootstrap_jitter": 0.05,
            "strict_refit": True,
            "peaks": [object()],
        },
        n_boot=16,
        seed=123,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    d = out.diagnostics
    assert d.get("bootstrap_mode") == "refit"
    assert d.get("jitter_applied_any") is True
    assert d.get("jitter_free_params", 0) >= 1
    assert calls["n"] >= 1
