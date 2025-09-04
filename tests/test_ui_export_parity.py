import numpy as np
from core.uncertainty import bootstrap_ci
from core.data_io import normalize_unc_result
from core.residuals import jacobian_fd


def pseudo_voigt(x, height, x0, fwhm, eta):
    eta = np.clip(eta, 0.0, 1.0)
    ga = np.exp(-4.0 * np.log(2.0) * ((x - x0) ** 2) / (fwhm ** 2))
    lo = 1.0 / (1.0 + 4.0 * ((x - x0) ** 2) / (fwhm ** 2))
    return height * ((1.0 - eta) * ga + eta * lo)


def test_ui_export_parity_smoke():
    x = np.linspace(-1, 1, 25)
    theta = np.array([0.0, 1.0, 1.0, 0.5])

    def predict_full(th):
        return pseudo_voigt(x, th[1], th[0], th[2], th[3])

    y = predict_full(theta)

    def resid_fn(th):
        return y - predict_full(th)

    r0 = resid_fn(theta)
    J = jacobian_fd(resid_fn, theta)

    fit_ctx = {
        "x": x,
        "y": y,
        "baseline": None,
        "mode": "add",
        "residual_fn": resid_fn,
        "predict_full": predict_full,
        "x_all": x,
    }

    out = bootstrap_ci(theta=theta, residual=r0, jacobian=J, predict_full=predict_full, fit_ctx=fit_ctx, n_boot=8)
    norm = normalize_unc_result(out)

    rmse = float(np.sqrt(np.mean(r0 ** 2)))
    dof = max(1, r0.size - theta.size)
    if not np.isfinite(norm.get("rmse", float("nan"))):
        norm["rmse"] = rmse
    if not np.isfinite(norm.get("dof", float("nan"))):
        norm["dof"] = dof

    assert "stats" in norm and isinstance(norm["stats"], list)
    assert "param_stats" in norm and all(k in norm["param_stats"] for k in ("center", "height", "fwhm", "eta"))
    assert "rmse" in norm and isinstance(norm["rmse"], float)
    assert "dof" in norm and isinstance(norm["dof"], (int, float)) and int(norm["dof"]) >= 1
