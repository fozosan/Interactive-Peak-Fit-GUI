import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic


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
    fit_peaks = [Peak(p.center, p.height, p.fwhm, p.eta) for p in start_peaks]

    state = classic.prepare_state(x, y, start_peaks, mode="subtract", baseline=None, opts={})["state"]
    for _ in range(20):
        state, _, _, _, _ = classic.iterate(state)

    model_step = pv_sum(x, state["peaks"])
    rmse_step = np.sqrt(np.mean((model_step - y) ** 2))

    res = classic.solve(x, y, fit_peaks, "subtract", None, {})
    rmse_fit = res["rmse"]

    assert rmse_step <= 5.0 * rmse_fit
