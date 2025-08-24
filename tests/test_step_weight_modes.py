import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit.bounds import pack_theta_bounds
from fit import step_engine


def test_step_weight_modes_smoke():
    x = np.linspace(-5, 5, 100)
    peaks = [Peak(0.0, 1.0, 1.0, 0.5)]
    y = pv_sum(x, peaks)
    _, bounds = pack_theta_bounds(peaks, x, {})
    wmin_eval = bounds[0][2] if bounds[0].size >= 3 else 1e-6
    for mode in ["none", "poisson", "inv_y"]:
        theta, cost, step_norm, accepted = step_engine.step_once(
            x,
            y,
            peaks,
            "subtract",
            None,
            loss="linear",
            weight_mode=mode,
            damping=0.0,
            trust_radius=np.inf,
            bounds=bounds,
            wmin_eval=wmin_eval,
            f_scale=1.0,
        )
        assert accepted and np.isfinite(cost)
