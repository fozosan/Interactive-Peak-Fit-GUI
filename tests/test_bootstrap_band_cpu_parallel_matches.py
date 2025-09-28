import numpy as np

from core.uncertainty import bootstrap_ci


def test_band_cpu_parallel_matches_sequential():
    x = np.linspace(0, 1, 200)
    theta = np.array([0.2, 0.5, 0.1, 0.3])
    y = np.sin(2 * np.pi * x)
    resid = y - y.mean()
    J = np.ones((x.size, theta.size))

    def refit(th0, locked_mask, bounds, x_in, y_in):
        return np.asarray(th0, float), True

    def model(th):
        return np.sin(2 * np.pi * x)

    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "baseline": None,
        "mode": "add",
        "refit": refit,
        "strict_refit": True,
        "bootstrap_jitter": 0.05,
        "peaks": [object()],
    }

    r_seq = bootstrap_ci(
        theta,
        resid,
        J,
        predict_full=model,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=64,
        seed=7,
        workers=0,
        alpha=0.1,
        center_residuals=True,
        return_band=True,
    )

    fit_ctx_p = dict(fit_ctx)
    fit_ctx_p["unc_band_workers"] = 4
    r_par = bootstrap_ci(
        theta,
        resid,
        J,
        predict_full=model,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx_p,
        n_boot=64,
        seed=7,
        workers=0,
        alpha=0.1,
        center_residuals=True,
        return_band=True,
    )

    assert r_seq.band is not None and r_par.band is not None
    _, lo_seq, hi_seq = r_seq.band
    _, lo_par, hi_par = r_par.band
    np.testing.assert_allclose(lo_seq, lo_par, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hi_seq, hi_par, rtol=1e-12, atol=1e-12)
    assert r_par.diagnostics.get("band_workers_used") == 4
