import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import iterate


def test_iterate_dispatch_cost_decreases():
    x = np.linspace(-5, 5, 100)
    true = [Peak(-1.0, 2.0, 1.0, 0.5)]
    y = pv_sum(x, true)
    peaks = [Peak(-0.5, 1.5, 1.2, 0.5)]

    options = {"solver": "classic"}
    state = {
        "x_fit": x,
        "y_fit": y,
        "peaks": peaks,
        "mode": "subtract",
        "baseline": None,
        "options": options,
    }

    model0 = pv_sum(x, peaks)
    cost0 = 0.5 * float(np.sum((model0 - y) ** 2))

    res_state, ok, c0, c1, _ = iterate(state)
    assert ok
    assert c1 <= cost0
