import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from fit import classic

def test_classic_step_vs_solve():
    x = np.linspace(0, 60, 400)
    true = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
    y = pv_sum(x, true)
    start = [Peak(19, 4, 6, 0.5), Peak(41, 1.5, 7, 0.3)]
    solve_res = classic.solve(
        x,
        y,
        [Peak(p.center, p.height, p.fwhm, p.eta) for p in start],
        mode="add",
        baseline=None,
        opts={},
    )
    state = classic.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(20):
        state, ok, c0, c1, info = classic.iterate(state)
    r = state["residual"](state["theta_free"])
    rmse_step = float(np.sqrt(np.mean(r**2)))
    target = max(solve_res["rmse"], 1e-8)
    assert rmse_step <= 1.01 * target
