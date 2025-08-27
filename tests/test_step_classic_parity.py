import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from core.peaks import Peak
from core.models import pv_sum
from core.fit_api import classic_step, build_residual_and_jacobian
from fit.bounds import pack_theta_bounds


def _flatten(peaks):
    arr = []
    for p in peaks:
        arr.extend([p.center, p.height, p.fwhm, p.eta])
    return np.asarray(arr, float)


def test_classic_step_parity():
    x = np.linspace(0.0, 60.0, 200)
    true = [Peak(20, 5, 5, 0.5), Peak(25, 2, 5, 0.3)]
    y = pv_sum(x, true)
    start = [Peak(19, 4, 6, 0.5, True), Peak(26, 1.5, 6, 0.3)]
    payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}

    theta0_full, (lo, hi) = pack_theta_bounds(start, x, {})
    theta1, res = classic_step(payload)
    assert res.accepted and res.cost1 < res.cost0
    assert res.step_norm > 0 and np.isfinite(res.cost1)
    assert np.all(theta1 >= lo - 1e-12) and np.all(theta1 <= hi + 1e-12)
    assert theta1[0] == pytest.approx(theta0_full[0])

    info = build_residual_and_jacobian(payload, "classic")
    theta = info["theta"]
    r0, J0 = info["residual_jac"](theta)
    lam = 1.0
    free = ~info["locked_mask"]
    delta_f = np.linalg.solve(J0[:, free].T @ J0[:, free] + lam * np.eye(free.sum()), -J0[:, free].T @ r0)
    delta_ref = np.zeros_like(theta)
    delta_ref[free] = delta_f
    delta_step = theta1 - theta
    cos = float(np.dot(delta_step, delta_ref) / (np.linalg.norm(delta_step) * np.linalg.norm(delta_ref)))
    assert cos >= 0.95

    # window parity
    mask = x < 40
    payload_win = {"x": x[mask], "y": y[mask], "peaks": start, "mode": "add", "baseline": None, "options": {}}
    info_win = build_residual_and_jacobian(payload_win, "classic")
    r_win, _ = info_win["residual_jac"](info_win["theta"])
    cost_win = 0.5 * float(r_win @ r_win)
    _, res_win = classic_step(payload_win)
    assert res_win.cost0 == pytest.approx(cost_win)

    # baseline parity
    base = np.full_like(x, 0.1)
    y_plus = y + base
    payload_base = {"x": x, "y": y_plus, "peaks": start, "mode": "add", "baseline": base, "options": {}}
    info_base = build_residual_and_jacobian(payload_base, "classic")
    r_base, _ = info_base["residual_jac"](info_base["theta"])
    cost_base = 0.5 * float(r_base @ r_base)
    _, res_base = classic_step(payload_base)
    assert res_base.cost0 == pytest.approx(cost_base)
