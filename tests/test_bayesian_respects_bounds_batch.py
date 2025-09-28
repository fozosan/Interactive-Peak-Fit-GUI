import numpy as np
from core.uncertainty import bayesian_ci


def test_bayesian_uses_bounds_from_fit_ctx_or_arg():
    x = np.linspace(0, 1, 50)
    center = 0.5
    theta_hat = np.array([center])
    y = np.sin(2 * np.pi * (x - center))

    def model(th):
        return np.sin(2 * np.pi * (x - th[0]))

    resid_fn = lambda th: y - model(th)

    lo = np.array([0.49])
    hi = np.array([0.51])
    locked = np.array([False])

    res = bayesian_ci(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=x,
        y_all=y,
        residual_fn=resid_fn,
        bounds=(lo, hi),
        locked_mask=locked,
        fit_ctx={"bounds": (lo, hi), "locked_mask": locked},
        return_band=False,
        n_steps=500,
        n_burn=200,
        param_names=["center"],
    )
    block = res.stats.get("p0") or {}
    est_arr = np.asarray(block.get("est", [np.nan])).flatten()
    assert est_arr.size > 0
    est = float(est_arr[0])
    assert est >= lo[0] - 1e-6 and est <= hi[0] + 1e-6
