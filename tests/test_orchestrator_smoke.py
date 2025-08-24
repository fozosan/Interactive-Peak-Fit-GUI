import importlib
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import models, peaks
from fit import orchestrator


def synthetic_data():
    x = np.linspace(-5.0, 5.0, 200)
    true = [
        peaks.Peak(-1.0, 1.0, 1.0, 0.3),
        peaks.Peak(1.2, 0.7, 1.1, 0.5),
    ]
    y = models.pv_sum(x, true)
    return x, y, true


def test_fallback_solvers_smoke():
    x, y, true = synthetic_data()
    solvers = ["classic", "modern_vp", "modern_trf"]
    if importlib.util.find_spec("lmfit") is not None:
        solvers.append("lmfit_vp")
    for name in solvers:
        init = [peaks.Peak(p.center + 0.2, p.height * 0.8, p.fwhm * 1.2, p.eta) for p in true]
        res = orchestrator.run_fit_with_fallbacks(x, y, init, "subtract", None, {"solver": name})
        assert res.success
        assert np.isfinite(res.rmse)
        assert len(res.peaks_out) == len(true)
