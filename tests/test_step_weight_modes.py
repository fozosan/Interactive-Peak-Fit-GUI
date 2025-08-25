import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic


def test_step_weight_modes():
    x = np.linspace(-5, 5, 100)
    true = [Peak(0.0, 1.0, 1.0, 0.5)]
    y = pv_sum(x, true)
    start = [Peak(0.1, 0.9, 1.1, 0.5)]
    for mode in ["none", "poisson", "inv_y"]:
        state = classic.prepare_state(x, y, start, mode="subtract", baseline=None, opts={"weights": mode})["state"]
        r = pv_sum(x, state["peaks"]) - y
        cost_fit = 0.5 * float(r @ r)
        _, _, c0, _, _ = classic.iterate(state)
        assert np.isclose(c0, cost_fit, rtol=1e-9, atol=1e-12)
