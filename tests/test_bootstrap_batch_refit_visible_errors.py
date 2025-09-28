import numpy as np

from core.uncertainty import bootstrap_ci


def test_batch_refit_errors_exposed():
    x = np.linspace(0, 1, 64)
    theta = np.array([0.3, 1.0, 0.2, 0.5])
    y = np.sin(2 * np.pi * (x - theta[0]))
    resid = y - y
    J = np.ones((x.size, theta.size))

    calls = {"n": 0}

    def bad_refit(theta_init, locked_mask, bounds, x_in, y_in):
        calls["n"] += 1
        if calls["n"] <= 2:
            return np.array(theta_init, float), True
        raise RuntimeError("boom")

    out = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx={
            "refit": bad_refit,
            "bootstrap_jitter": 0.02,
            "strict_refit": True,
            "peaks": [object()],
        },
        n_boot=4,
        seed=0,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    d = out.diagnostics
    assert isinstance(d.get("refit_errors"), list)
    assert any("boom" in str(s) for s in d.get("refit_errors"))
    assert calls["n"] >= 2
