import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from core.peaks import Peak
from core.models import pv_sum
from core.fit_api import modern_trf_step, modern_vp_step, lmfit_step
from fit.bounds import pack_theta_bounds
from fit import modern, modern_vp, lmfit_backend
from core.residuals import build_residual_jac
from core.weights import noise_weights
from fit.utils import robust_cost
from scipy.optimize import least_squares


def _problem():
    x = np.linspace(0.0, 60.0, 200)
    true = [Peak(20, 5, 5, 0.5), Peak(25, 2, 5, 0.3)]
    y = pv_sum(x, true)
    start = [Peak(19, 4, 6, 0.5, True), Peak(26, 1.5, 6, 0.3)]
    return x, y, start


def test_trf_step_parity():
    x, y, start = _problem()
    payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}
    theta_step, res = modern_trf_step(payload)
    assert res.accepted and res.cost1 < res.cost0 and res.step_norm > 0
    theta0_full, bounds_full = pack_theta_bounds(start, x, {})
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(1e-6, 2.0 * dx_med)
    theta0, bounds, x_scale, indices = modern._to_solver_vectors(theta0_full, bounds_full, start, fwhm_min)
    weights = noise_weights(y, "none")
    resid_jac = build_residual_jac(x, y, start, "add", None, weights)
    def fun(t):
        return resid_jac(t)[0]
    def jac(t):
        return resid_jac(t)[1]
    ref = least_squares(fun, theta0, jac=jac, method="trf", loss="linear", f_scale=1.0, bounds=bounds, x_scale=x_scale, max_nfev=2)
    theta_ref = theta0_full.copy()
    theta_ref[indices] = ref.x
    assert np.allclose(theta_step, theta_ref, rtol=1e-6)
    assert np.isfinite(res.cost1)


def test_vp_step_parity():
    x, y, start = _problem()
    payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}
    theta_step, res = modern_vp_step(payload)
    assert res.accepted and res.cost1 < res.cost0 and res.step_norm > 0
    ref = modern_vp.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {"maxfev": 1})
    assert np.allclose(theta_step, ref["theta"], rtol=1e-6)
    assert np.isfinite(res.cost1)


def test_lmfit_step_parity():
    try:
        import lmfit  # noqa: F401
    except Exception:
        pytest.skip("lmfit not installed")
    x, y, start = _problem()
    payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}
    theta_step, res = lmfit_step(payload)
    assert res.accepted and res.cost1 < res.cost0 and res.step_norm > 0
    ref = lmfit_backend.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {"maxfev": 1})
    assert np.allclose(theta_step, ref["theta"], rtol=1e-6)
    assert np.isfinite(res.cost1)
