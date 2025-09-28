import numpy as np

from core.uncertainty import bayesian_ci


def test_bayesian_posterior_health_fields_present():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 64)
    theta_true = np.array([0.5])

    def model(th):
        return np.sin(2 * np.pi * (x - th[0]))

    y = model(theta_true) + rng.normal(0, 0.05, size=x.size)

    theta_hat = np.array([0.48])
    bounds = (np.array([0.3]), np.array([0.7]))
    locked_mask = np.array([False])

    res = bayesian_ci(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=lambda th: y - model(th),
        bounds=bounds,
        locked_mask=locked_mask,
        fit_ctx={"bayes_burn": 300, "bayes_steps": 600, "bayes_thin": 2},
        return_band=False,
        seed=123,
    )
    d = res.diagnostics
    assert "ess_min" in d and "rhat_max" in d and "mcse_mean" in d
