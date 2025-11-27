import numpy as np
from core.models import pv_sum
from core.peaks import Peak
from core.uncertainty import bootstrap_ci
from fit.bounds import pack_theta_bounds


def test_bootstrap_refits_respect_width_caps():
    x = np.linspace(0, 100, 401)
    true = [Peak(50.0, 5.0, 10.0, 0.5)]
    y = pv_sum(x, true)
    theta0 = np.array([50.0, 4.5, 12.0, 0.5], float)
    opts = {"width_caps": [6.0]}
    _, bounds = pack_theta_bounds([Peak(50.0, 4.5, 12.0, 0.5)], x, opts)

    def predict_full(th):
        c, h, w, e = th.reshape(-1)
        return pv_sum(x, [Peak(c, h, w, e)])

    r0 = predict_full(theta0) - y
    J = np.zeros((x.size, theta0.size))
    fit_ctx = {
        "x": x, "y": y, "mode": "add",
        "solver": "modern_vp",
        "solver_options": opts,
        "peaks": [Peak(50.0, 4.5, 12.0, 0.5)],
        "strict_refit": True,
        "unc_workers": 0,
    }
    out = bootstrap_ci(
        theta=theta0,
        residual=r0,
        jacobian=J,
        predict_full=predict_full,
        x_all=x, y_all=y,
        fit_ctx=fit_ctx,
        bounds=bounds,
        param_names=["center", "height", "fwhm", "eta"],
        n_boot=8,
        seed=0, workers=0,
    )
    stats = out.stats.get("fwhm", {})
    q975 = float(
        stats.get("p97.5",
        stats.get("q97.5",
        stats.get("q95", np.nan)))
    )
    assert np.isfinite(q975) and q975 <= 6.0 + 1e-9
