import numpy as np
from core import fit_api, uncertainty


def test_bootstrap_basic(two_peak_data, rng, monkeypatch):
    noisy = dict(two_peak_data)
    noisy["y"] = noisy["y"] + 0.01 * rng.standard_normal(noisy["x"].size)
    fit = fit_api.run_fit_consistent(
        **noisy, return_jacobian=True, return_predictors=True
    )

    def fake_run_fit_consistent(x, y, cfg, theta_init=None, locked_mask=None, bounds=None, baseline=None):
        locked_mask = np.asarray(locked_mask, bool)
        th = np.asarray(theta_init, float).copy()
        mean_y = float(np.mean(y))
        th[~locked_mask] += mean_y * 0.01
        return {"fit_ok": True, "theta": th}

    monkeypatch.setattr(fit_api, "run_fit_consistent", fake_run_fit_consistent)

    args = dict(
        theta=fit["theta"],
        residual=fit["residual"],
        jacobian=fit["jacobian"],
        predict_full=fit["predict_full"],
        x_all=fit["x"],
        y_all=noisy["y"],
        bounds=fit.get("bounds"),
        param_names=fit.get("param_names"),
        locked_mask=fit.get("locked_mask"),
        fit_ctx=fit,
        n_boot=30,
        seed=123,
        return_band=True,
    )

    res1 = uncertainty.bootstrap_ci(**args)
    assert res1.band is not None
    assert np.all(np.isfinite(res1["param_std"]))
    assert np.all(res1["param_std"] >= 0)
    assert np.any(res1["param_std"] > 0)

    res2 = uncertainty.bootstrap_ci(**args)
    assert np.allclose(res1["param_mean"], res2["param_mean"])
    assert np.allclose(res1["param_std"], res2["param_std"])
