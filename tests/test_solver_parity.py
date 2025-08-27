import sys
import pathlib
import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from infra import performance
from core.peaks import Peak

pytestmark = pytest.mark.filterwarnings("ignore:.*")


def _reference_total(x, peaks):
    y = np.zeros_like(x, dtype=float)
    for h, c, w, eta in peaks:
        w = max(w, 1e-12)
        eta = min(1.0, max(0.0, eta))
        dx = (x - c) / w
        ga = np.exp(-4.0 * np.log(2.0) * dx * dx)
        lo = 1.0 / (1.0 + 4.0 * dx * dx)
        y += h * ((1.0 - eta) * ga + eta * lo)
    return y


def _rmse(theta, x, y):
    n = theta.size // 4
    pk = [(theta[4*i+1], theta[4*i+0], theta[4*i+2], theta[4*i+3]) for i in range(n)]
    y_fit = _reference_total(x, pk)
    return float(np.sqrt(np.mean((y_fit - y) ** 2)))


@pytest.mark.parametrize("solver", ["classic", "modern_trf", "modern_vp"])
def test_solver_parity(solver):
    performance.set_gpu(False)
    performance.set_numba(False)
    x = np.linspace(-1, 1, 100)
    true_peaks = [
        Peak(-0.2, 1.0, 0.3, 0.2),
        Peak(0.0, 0.8, 0.25, 0.5),
        Peak(0.15, 0.6, 0.2, 0.3),
    ]
    y = _reference_total(x, [(p.height, p.center, p.fwhm, p.eta) for p in true_peaks])
    init = [Peak(p.center, p.height, p.fwhm, p.eta) for p in true_peaks]
    if solver == "classic":
        from fit.classic import solve
    elif solver == "modern_trf":
        from fit.modern import solve
    else:
        from fit.modern_vp import solve
    res1 = solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in init], "add", None, {})
    theta1 = np.asarray(res1["theta"], dtype=float)
    rmse1 = _rmse(theta1, x, y)
    performance.set_numba(True)
    res2 = solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in init], "add", None, {})
    theta2 = np.asarray(res2["theta"], dtype=float)
    rmse2 = _rmse(theta2, x, y)
    assert np.allclose(theta1, theta2, rtol=5e-8, atol=1e-10)
    assert abs(rmse1 - rmse2) <= 1e-10
    performance.set_numba(False)
