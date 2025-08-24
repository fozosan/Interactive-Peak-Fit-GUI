import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit import modern
from core.peaks import Peak
from core.models import pv_sum


def test_modern_subtract_uses_baseline():
    x = np.linspace(0, 10, 100)
    baseline = 0.1 * x
    peak_true = Peak(5.0, 2.0, 1.0, 0.5)
    y = pv_sum(x, [peak_true]) + baseline
    guess = [Peak(5.0, 1.0, 1.0, 0.5)]

    res = modern.solve(x, y, guess, mode="subtract", baseline=baseline, options={})

    assert np.allclose(res["theta"], [5.0, 2.0, 1.0, 0.5])
