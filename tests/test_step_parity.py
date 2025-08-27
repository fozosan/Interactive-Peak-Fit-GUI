import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from core.peaks import Peak
from core.models import pv_sum
from core.fit_api import build_residual_and_jacobian


def _scenario(overlap: bool):
    x = np.linspace(0.0, 60.0, 400)
    if overlap:
        true = [Peak(20, 5, 5, 0.5), Peak(25, 2, 5, 0.3)]
        start = [Peak(19, 4, 6, 0.5, True), Peak(26, 1.5, 6, 0.3)]
    else:
        true = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
        start = [Peak(19, 4, 6, 0.5, True), Peak(41, 1.5, 7, 0.3)]
    y = pv_sum(x, true)
    return x, y, start


def _gn_step(payload, solver):
    info = build_residual_and_jacobian(payload, solver)
    theta0 = info["theta"]
    r0, J0 = info["residual_jac"](theta0)
    cost0 = 0.5 * float(r0 @ r0)
    free = ~info["locked_mask"]
    Jf = J0[:, free]
    lam = 1.0
    delta_f = np.linalg.solve(Jf.T @ Jf + lam * np.eye(Jf.shape[1]), -Jf.T @ r0)
    delta = np.zeros_like(theta0)
    delta[free] = delta_f
    theta1 = np.clip(theta0 + delta, info["bounds"][0], info["bounds"][1])
    r1, _ = info["residual_jac"](theta1)
    cost1 = 0.5 * float(r1 @ r1)
    return theta0, theta1, delta, cost0, cost1, info


def test_step_parity():
    lmfit_available = True
    try:  # optional dependency
        import lmfit  # noqa: F401
    except Exception:  # pragma: no cover - dependency may be missing
        lmfit_available = False

    for overlap in (False, True):
        x, y, start = _scenario(overlap)
        payload = {"x": x, "y": y, "peaks": start, "mode": "add", "baseline": None, "options": {}}

        th_c0, th_c1, d_c, c0, c1, info_c = _gn_step(payload, "classic")
        assert c1 < c0

        th_m0, th_m1, d_m, cm0, cm1, info_m = _gn_step(payload, "modern_trf")
        assert cm1 < cm0

        if not overlap:
            dot = float(np.dot(d_m, d_c))
            cos = dot / (np.linalg.norm(d_m) * np.linalg.norm(d_c))
            assert dot > 0 and cos > 0.95

        if lmfit_available:
            th_l0, th_l1, d_l, cl0, cl1, info_l = _gn_step(payload, "lmfit_vp")
            assert cl1 < cl0
            lo, hi = info_l["bounds"]
            assert np.all(th_l1 >= lo - 1e-12) and np.all(th_l1 <= hi + 1e-12)
            locked = info_l["locked_mask"]
            assert np.all(th_l1[locked] == th_l0[locked])

