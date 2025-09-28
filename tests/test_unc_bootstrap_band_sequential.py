import numpy as np

from core.uncertainty import bootstrap_ci


def test_bootstrap_band_sequential_no_threadpool():
    """Bootstrap band evaluation should run sequentially without workers."""
    x = np.linspace(0, 1, 50)
    theta = np.array([1.0, 2.0])
    yhat = theta[0] + theta[1] * x
    rng = np.random.default_rng(0)
    y = yhat + rng.normal(0, 0.1, size=x.size)
    resid = y - yhat
    J = np.vstack([np.ones_like(x), x]).T
    fit_ctx = {"x_all": x, "y_all": y, "baseline": None, "mode": "add"}

    res = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: th[0] + th[1] * x,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=32,
        seed=None,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=True,
    )

    assert res.band is not None
    assert res.diagnostics.get("workers_used", None) is None
