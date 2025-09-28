import numpy as np
import types

from core import fit_api
from core.uncertainty import bootstrap_ci


def test_jitter_disables_linear_and_calls_refit():
    """Percent jitter routed through fit_ctx should trigger refit attempts."""
    x = np.linspace(0, 1, 40)
    theta0 = np.array([0.5, 1.5, 0.2, 0.5])
    rng = np.random.default_rng(0)
    y = np.sin(2 * np.pi * x) + 0.01 * rng.normal(size=x.size)
    resid = y - y.mean()
    J = np.ones((x.size, theta0.size))

    called = {"count": 0}

    def fake_run_fit_consistent(*, x, y, cfg=None, config=None, peaks=None, peaks_in=None,
                                 theta_init=None, locked_mask=None, bounds=None, **kwargs):
        called["count"] += 1
        theta_init = np.asarray(theta_init, float)
        return {"theta": theta_init, "fit_ok": True}

    peak = types.SimpleNamespace(
        center=0.25,
        height=1.0,
        fwhm=0.1,
        eta=0.5,
        lock_center=False,
        lock_width=False,
    )
    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "baseline": None,
        "mode": "add",
        "peaks": [peak],
        "solver": "classic",
        "bootstrap_jitter": 5.0,  # percent; expect normalization and refit path
        "allow_linear_fallback": True,
    }

    orig = fit_api.run_fit_consistent
    try:
        fit_api.run_fit_consistent = fake_run_fit_consistent  # type: ignore[assignment]
        out = bootstrap_ci(
            theta=theta0,
            residual=resid,
            jacobian=J,
            predict_full=lambda th: y,
            x_all=x,
            y_all=y,
            fit_ctx=fit_ctx,
            n_boot=16,
            seed=0,
            workers=None,
            alpha=0.1,
            center_residuals=True,
            return_band=False,
        )
    finally:
        fit_api.run_fit_consistent = orig  # type: ignore[assignment]

    assert out.diagnostics.get("bootstrap_mode") == "refit"
    assert called["count"] >= 1
