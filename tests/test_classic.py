import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit import classic
from core.peaks import Peak


def test_classic_clips_parameters():
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    peaks = [Peak(-5.0, -1.0, -2.0, 1.5)]
    res = classic.solve(
        x,
        y,
        peaks,
        mode="add",
        baseline=None,
        options={"centers_in_window": True, "min_fwhm": 1.0},
    )
    theta = res["theta"]
    assert res["ok"]
    assert np.allclose(theta, [0.0, 0.0, 1.0, 1.0])
