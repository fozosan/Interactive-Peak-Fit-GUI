import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.peaks import Peak
from fit import modern, step_engine
from fit.bounds import pack_theta_bounds


def test_modern_respects_locks_and_bounds():
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    peaks = [Peak(5.0, -1.0, -0.5, 1.2, lock_center=True, lock_width=True)]
    res = modern.solve(
        x,
        y,
        peaks,
        mode="add",
        baseline=None,
        options={"centers_in_window": True, "min_fwhm": 1.0},
    )
    theta = res["theta"]
    assert np.allclose(theta, [5.0, 0.0, 1.0, 1.0])


def test_step_engine_respects_bounds():
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    peaks = [Peak(5.0, -1.0, -0.5, 1.2, lock_center=True, lock_width=True)]
    _, bounds = pack_theta_bounds(peaks, x, {"centers_in_window": True, "min_fwhm": 1.0})
    theta, _ = step_engine.step_once(
        x,
        y,
        peaks,
        "add",
        None,
        loss="linear",
        weights=None,
        damping=0.0,
        trust_radius=np.inf,
        bounds=bounds,
    )
    assert np.allclose(theta, [5.0, 0.0, 1.0, 1.0])
