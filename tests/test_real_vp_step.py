import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import modern_vp


def _load(name):
    data = np.loadtxt(Path(__file__).parent / "fixtures" / name, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def test_modern_vp_step_real():
    cases = [
        ("real1.csv", [Peak(14, 7, 5, 0.5), Peak(34, 4, 7, 0.3)]),
        (
            "real2.csv",
            [Peak(9, 2.5, 5, 0.4), Peak(24, 3.5, 6, 0.5), Peak(39, 5, 4, 0.2)],
        ),
    ]
    for fname, start in cases:
        x, y = _load(fname)
        full = modern_vp.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {})
        theta_full = full["theta"]
        peaks_full = [
            Peak(theta_full[4 * i], theta_full[4 * i + 1], theta_full[4 * i + 2], theta_full[4 * i + 3])
            for i in range(len(start))
        ]
        rmse_full = np.sqrt(np.mean((pv_sum(x, peaks_full) - y) ** 2))
        s = modern_vp.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
        for _ in range(25):
            s, ok, c0, c1, info = modern_vp.iterate(s)
        theta = s["theta"]
        peaks_final = [
            Peak(theta[4 * i], theta[4 * i + 1], theta[4 * i + 2], theta[4 * i + 3])
            for i in range(len(start))
        ]
        rmse_step = np.sqrt(np.mean((pv_sum(x, peaks_final) - y) ** 2))
        assert rmse_step <= 1.05 * rmse_full

