import pathlib, sys

import numpy as np
import pytest
from scipy.optimize import least_squares

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from core.fit_api import step_modern_trf, step_modern_vp, step_lmfit_vp
from fit.bounds import pack_theta_bounds
from fit.modern import _to_solver_vectors
from core.residuals import build_residual_jac
from core.weights import noise_weights
from fit.utils import robust_cost


def _scenario():
    x = np.linspace(0.0, 50.0, 200)
    true = [Peak(15, 4, 4, 0.4), Peak(28, 2, 5, 0.3)]
    start = [Peak(14, 3, 5, 0.4, True), Peak(29, 1.5, 6, 0.3)]
    y = pv_sum(x, true)
    return x, y, start


def test_step_parity_trf():
    x, y, start = _scenario()
    theta_step, peaks_step, res = step_modern_trf(x, y, start, "subtract", None, {})
    assert res.accepted
    assert res.cost1 < res.cost0

    weights = noise_weights(y, "none")
    theta0_full, bounds_full = pack_theta_bounds(start, x, {})
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(1e-6, 2.0 * dx_med)
    theta0, bounds, x_scale, idx = _to_solver_vectors(theta0_full, bounds_full, start, fwhm_min)
    rj = build_residual_jac(x, y, start, "subtract", None, weights)

    def fun(t):
        r, _ = rj(t)
        return r

    def jac(t):
        _, J = rj(t)
        return J

    res_ref = least_squares(
        fun,
        theta0,
        jac=jac,
        method="trf",
        loss="linear",
        f_scale=1.0,
        bounds=bounds,
        x_scale=x_scale,
        max_nfev=2,
    )
    theta_ref = theta0_full.copy()
    theta_ref[idx] = res_ref.x
    assert np.allclose(theta_step, theta_ref, rtol=1e-6, atol=1e-8)
    lo, hi = bounds_full
    assert np.all(theta_step >= lo - 1e-12) and np.all(theta_step <= hi + 1e-12)


def test_step_parity_vp():
    x, y, start = _scenario()
    theta_step, peaks_step, res = step_modern_vp(x, y, start, "subtract", None, {})
    assert res.accepted
    assert res.cost1 < res.cost0

    from fit import modern_vp

    prep = modern_vp.prepare_state(x, y, start, "subtract", None, {})["state"]
    state_ref, ok, c0, c1, info = modern_vp.iterate(prep)
    theta_ref = state_ref["theta"]
    assert np.allclose(theta_step, theta_ref, rtol=1e-6, atol=1e-8)


def test_step_parity_lmfit():
    try:
        import lmfit  # noqa: F401
    except Exception:
        pytest.skip("lmfit not installed")

    x, y, start = _scenario()
    theta_step, peaks_step, res = step_lmfit_vp(x, y, start, "subtract", None, {})
    assert res.accepted
    assert res.cost1 < res.cost0

    from fit import lmfit_backend

    ref = lmfit_backend.solve(x, y, start, "subtract", None, {"maxfev": 1})
    theta_ref = ref["theta"]
    assert np.allclose(theta_step, theta_ref, rtol=1e-6, atol=1e-8)
