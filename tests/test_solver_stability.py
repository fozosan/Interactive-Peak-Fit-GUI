import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core import models, peaks
from fit import orchestrator


def test_modern_and_lmfit_stable():
    x = np.linspace(-5.0, 5.0, 200)
    true = [peaks.Peak(-1.0, 1.0, 1.0, 0.3), peaks.Peak(1.2, 0.7, 1.1, 0.5)]
    y = models.pv_sum(x, true)
    init = [peaks.Peak(p.center + 0.2, p.height * 0.8, p.fwhm * 1.2, p.eta) for p in true]
    expected = np.array([-1.0, 1.0, 1.0, 0.3, 1.2, 0.7, 1.1, 0.5])
    solvers = ["modern_vp", "modern_trf"]
    try:
        import lmfit  # noqa: F401
        solvers.append("lmfit_vp")
    except Exception:
        pass
    for name in solvers:
        res = orchestrator.run_fit_with_fallbacks(x, y, init, "subtract", None, {"solver": name})
        assert res.solver == name
        assert np.allclose(res.theta, expected, atol=1e-12)
        assert res.cost < 1e-12
