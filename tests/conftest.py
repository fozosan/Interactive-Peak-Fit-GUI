import os
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend when headless
HEADLESS = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if HEADLESS:
    matplotlib.use("Agg")

# ensure project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

# deterministic RNG fixture
@pytest.fixture
def rng():
    return np.random.default_rng(123)

# synthetic two-peak data fixture for core-level tests
@pytest.fixture
def two_peak_data(rng):
    from core import peaks, models

    x = np.linspace(-5.0, 5.0, 201)
    seeds = [
        peaks.Peak(-0.8, 1.0, 0.6, 0.5),
        peaks.Peak(0.9, 0.8, 0.5, 0.4),
    ]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_restarts": 1,
        "perf_seed_all": True,
    }
    data = {
        "x": x,
        "y": y,
        "peaks_in": seeds,
        "cfg": cfg,
        "baseline": None,
        "mode": "add",
        "fit_mask": mask,
        "rng_seed": 123,
    }
    return data

# marker for GUI dependent tests
skip_if_no_display = pytest.mark.skipif(HEADLESS, reason="requires display")

# numeric comparison helper

def close_to(a, b, rtol=5e-5, atol=5e-8):
    return np.allclose(a, b, rtol=rtol, atol=atol)

# helper to ensure CSVs have no blank lines

@pytest.fixture
def no_blank_lines():
    def _check(path: Path) -> bool:
        text = Path(path).read_text()
        return "\n\n" not in text
    return _check
