import pathlib, sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from core.fit_api import step_classic, build_residual_and_jacobian
from fit.bounds import pack_theta_bounds


def _gn_reference(payload):
    info = build_residual_and_jacobian(payload, "classic")
    th0 = info["theta"]
    r0, J0 = info["residual_jac"](th0)
    free = ~info["locked_mask"]
    Jf = J0[:, free]
    delta_f = np.linalg.solve(Jf.T @ Jf + np.eye(Jf.shape[1]), -Jf.T @ r0)
    delta = np.zeros_like(th0)
    delta[free] = delta_f
    th1 = np.clip(th0 + delta, info["bounds"][0], info["bounds"][1])
    return th0, th1, delta, 0.5 * float(r0 @ r0)


def test_classic_step_parity():
    x = np.linspace(0.0, 60.0, 400)
    true = [Peak(20, 5, 5, 0.5), Peak(25, 2, 5, 0.3)]
    start = [Peak(19, 4, 6, 0.5, True), Peak(26, 1.5, 6, 0.3)]
    y = pv_sum(x, true)

    payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}
    th0, th1_ref, delta_ref, cost0_ref = _gn_reference(payload)

    theta, peaks_out, res = step_classic(x, y, start, "add", None, {})

    assert res.accepted
    assert res.cost1 < res.cost0
    assert res.step_norm > 0
    assert np.all(np.isfinite(theta))

    theta0, bounds = pack_theta_bounds(start, x, {})
    lo, hi = bounds
    assert np.all(theta >= lo - 1e-12) and np.all(theta <= hi + 1e-12)
    assert theta[0] == pytest.approx(theta0[0])  # locked center

    delta_step = theta - th0
    cos = float(np.dot(delta_step, delta_ref)) / (
        np.linalg.norm(delta_step) * np.linalg.norm(delta_ref)
    )
    assert cos >= 0.95

    info = build_residual_and_jacobian(payload, "classic")
    r0, _ = info["residual_jac"](info["theta"])
    cost_ref = 0.5 * float(r0 @ r0)
    assert res.cost0 == pytest.approx(cost_ref)

    payload_sub = {"x": x, "y": y, "peaks": start, "mode": "subtract", "baseline": None, "options": {}}
    info_sub = build_residual_and_jacobian(payload_sub, "classic")
    r0s, _ = info_sub["residual_jac"](info_sub["theta"])
    cost_sub = 0.5 * float(r0s @ r0s)
    _, _, res_sub = step_classic(x, y, start, "subtract", None, {})
    assert res_sub.cost0 == pytest.approx(cost_sub)
