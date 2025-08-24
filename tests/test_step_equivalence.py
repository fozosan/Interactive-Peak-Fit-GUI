import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic, step_engine
from fit.bounds import pack_theta_bounds


def _update_peaks(peaks, theta):
    for i, p in enumerate(peaks):
        p.center = theta[4 * i + 0]
        p.height = theta[4 * i + 1]
        p.fwhm = theta[4 * i + 2]
        p.eta = theta[4 * i + 3]


def test_step_converges_close_to_fit():
    x = np.linspace(-5, 5, 200)
    true_peaks = [
        Peak(-1.0, 3.0, 1.2, 0.2),
        Peak(2.0, 2.0, 0.8, 0.3),
    ]
    y = pv_sum(x, true_peaks)

    start_peaks = [
        Peak(-0.8, 2.5, 1.0, 0.2),
        Peak(2.2, 1.5, 1.1, 0.3),
    ]
    step_peaks = [Peak(p.center, p.height, p.fwhm, p.eta) for p in start_peaks]
    fit_peaks = [Peak(p.center, p.height, p.fwhm, p.eta) for p in start_peaks]

    options = {}
    theta0, bounds = pack_theta_bounds(step_peaks, x, options)
    weight_mode = "none"  # no noise weighting

    model0 = pv_sum(x, step_peaks)
    r0 = model0 - y
    cost0 = 0.5 * float(np.dot(r0, r0))
    theta, cost1 = step_engine.step_once(
        x,
        y,
        step_peaks,
        "subtract",
        None,
        loss="linear",
        weight_mode=weight_mode,
        damping=0.0,
        trust_radius=np.inf,
        bounds=bounds,
        f_scale=1.0,
    )
    assert cost1 < cost0
    lb, ub = bounds
    assert np.all(theta >= lb) and np.all(theta <= ub)
    _update_peaks(step_peaks, theta)

    for _ in range(19):
        theta, cost1 = step_engine.step_once(
            x,
            y,
            step_peaks,
            "subtract",
            None,
            loss="linear",
            weight_mode=weight_mode,
            damping=0.0,
            trust_radius=np.inf,
            bounds=bounds,
            f_scale=1.0,
        )
        _update_peaks(step_peaks, theta)

    model_step = pv_sum(x, step_peaks)
    rmse_step = np.sqrt(np.mean((model_step - y) ** 2))

    res = classic.solve(x, y, fit_peaks, "subtract", None, options)
    theta_fit = res["theta"]
    fit_final = [
        Peak(theta_fit[4 * i + 0], theta_fit[4 * i + 1], theta_fit[4 * i + 2], theta_fit[4 * i + 3])
        for i in range(len(fit_peaks))
    ]
    model_fit = pv_sum(x, fit_final)
    rmse_fit = np.sqrt(np.mean((model_fit - y) ** 2))

    assert rmse_step <= 1.05 * rmse_fit
