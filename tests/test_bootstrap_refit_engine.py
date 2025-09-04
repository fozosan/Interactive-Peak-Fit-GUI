import numpy as np
import types
import pytest

from core.uncertainty import bootstrap_ci


def test_bootstrap_refit_seed_repro(monkeypatch):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    theta = np.array([1.0, 1.0, 1.0, 0.5])
    residual = y.copy()
    jac = np.eye(4)

    def fake_run_fit_consistent(x, y, cfg, theta_init=None, locked_mask=None, bounds=None, baseline=None):
        locked_mask = np.asarray(locked_mask, bool)
        th = np.asarray(theta_init, float).copy()
        mean_y = float(np.mean(y))
        th[~locked_mask] += mean_y * 0.01
        return {"fit_ok": True, "theta": th}

    monkeypatch.setattr("core.fit_api.run_fit_consistent", fake_run_fit_consistent)

    peak = types.SimpleNamespace(center=1.0, height=1.0, fwhm=1.0, eta=0.5, lock_center=False, lock_width=False)
    fit_ctx = {"peaks": [peak], "mode": "add", "baseline": np.zeros_like(x), "solver": "classic"}
    locked = np.array([True, False, False, False])

    args = dict(
        theta=theta,
        residual=residual,
        jacobian=jac,
        predict_full=lambda th, x=x: np.zeros_like(x),
        x_all=x,
        y_all=y,
        locked_mask=locked,
        fit_ctx=fit_ctx,
        n_boot=64,
        seed=123,
        return_band=True,
    )

    res1 = bootstrap_ci(**args)
    res2 = bootstrap_ci(**args)

    for k in res1.stats:
        for fld in ("est", "sd", "p2.5", "p97.5"):
            assert res1.stats[k][fld] == pytest.approx(res2.stats[k][fld])

    assert res1.stats["p0"]["sd"] == pytest.approx(0.0, abs=1e-12)
    assert res1.band is not None
