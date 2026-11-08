"""A bootstrap called without a solver refits with the app's default solver.

The GUI and batch always name the refit solver in ``fit_ctx``. Direct callers,
such as tests that pass a ``run_fit_consistent`` result as ``fit_ctx``, often do
not, and commit d8146b0 turned that case into a KeyError. The default is
``modern_vp``, the solver the GUI and batch pass when nothing else is configured.
"""
import numpy as np

from core import fit_api, uncertainty


def test_bootstrap_without_solver_refits_with_default(two_peak_data, monkeypatch):
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    fit_ctx = dict(fit)
    assert not {"solver", "unc_boot_solver", "solver_used"} & fit_ctx.keys()

    solvers_seen = []

    def spy_refit(x, y, cfg, theta_init=None, locked_mask=None, bounds=None, baseline=None):
        solvers_seen.append(cfg.get("solver"))
        start = fit["theta"] if theta_init is None else theta_init
        return {"fit_ok": True, "theta": np.asarray(start, float)}

    monkeypatch.setattr(fit_api, "run_fit_consistent", spy_refit)

    res = uncertainty.bootstrap_ci(
        theta=fit["theta"],
        residual=fit["residual"],
        jacobian=fit["jacobian"],
        predict_full=fit["predict_full"],
        x_all=fit["x"],
        y_all=two_peak_data["y"],
        fit_ctx=fit_ctx,
        n_boot=5,
        seed=0,
        workers=0,
    )

    assert solvers_seen, "the bootstrap made no refits"
    assert set(solvers_seen) == {"modern_vp"}
    assert res.diagnostics["solver_used"] == "modern_vp"
