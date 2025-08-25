import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import modern_vp, modern


def _run(backend):
    x = np.linspace(-5, 5, 200)
    true = [Peak(-1.0, 1.0, 1.0, 0.3), Peak(1.5, 0.8, 1.2, 0.4)]
    y = pv_sum(x, true)
    start = [Peak(p.center + 0.1, p.height * 0.9, p.fwhm * 1.1, p.eta) for p in true]
    state = backend.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    last = float("inf")
    for _ in range(30):
        state, ok, c0, c1, info = backend.iterate(state)
        assert np.isfinite(c1)
        assert c1 <= c0 + 1e-12
        assert c1 <= last + 1e-12
        last = c1


def test_modern_vp_no_freeze():
    _run(modern_vp)


def test_modern_trf_no_freeze():
    _run(modern)
