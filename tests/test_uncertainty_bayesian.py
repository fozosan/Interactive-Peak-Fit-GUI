import numpy as np
import pytest

from core.uncertainty_router import route_uncertainty
from tests.conftest import bayes_knobs, ensure_unc_common


def _toy_setup():
    x = np.linspace(0.0, 1.0, 8)
    theta = np.array([0.25, -0.5], float)
    def model_eval(th, xvals=x):
        return th[0] + th[1] * xvals
    y = model_eval(theta)
    def residual_fn(th, xvals=x, y_obs=y):
        return y_obs - model_eval(th, xvals=xvals)
    jacobian = np.vstack([np.ones_like(x), x]).T
    return theta, residual_fn, jacobian, model_eval, x, y


def test_route_uncertainty_bayesian_requires_knobs():
    pytest.importorskip("emcee")
    theta_hat, residual_fn, jac, model_eval, x, y = _toy_setup()

    ctx = ensure_unc_common({
        "alpha": 0.05,
        **bayes_knobs(walkers=0, burn=100, steps=200, thin=2),
    })

    res = route_uncertainty(
        "bayesian",
        theta_hat=theta_hat,
        residual_fn=residual_fn,
        jacobian=jac,
        model_eval=model_eval,
        fit_ctx=ctx,
        x_all=x,
        y_all=y,
        workers=0,
        seed=123,
    )

    assert res.method == "bayesian"
    assert res.stats
