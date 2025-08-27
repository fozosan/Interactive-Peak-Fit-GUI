import importlib
import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.filterwarnings("ignore:.*")


def _reload_performance(monkeypatch):
    import infra.performance as performance
    return importlib.reload(performance)


def test_shadow_compare_ok(monkeypatch):
    monkeypatch.delenv("GL_PERF_SHADOW_COMPARE", raising=False)
    performance = _reload_performance(monkeypatch)
    logs = []
    performance.set_logger(lambda m, lvl="INFO": logs.append((lvl, m)))
    performance.enable_shadow_compare(True)
    performance.set_numba(False)
    performance.set_gpu(False)
    x = np.linspace(-1, 1, 20)
    peaks = [(1.0, 0.0, 0.5, 0.3), (0.8, 0.4, 0.2, 0.5)]
    y = performance.eval_total(x, peaks)
    comps = performance.eval_components(x, peaks)
    assert np.allclose(comps.sum(axis=0), y)
    assert logs == []
    performance.enable_shadow_compare(False)
    performance.set_logger(None)


@pytest.mark.skipif(not getattr(importlib.import_module("infra.performance"), "_NUMBA_OK", False), reason="Numba not available")
def test_shadow_compare_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("GL_PERF_SHADOW_COMPARE", "1")
    performance = _reload_performance(monkeypatch)
    if not performance._NUMBA_OK:
        pytest.skip("Numba not available")
    logs = []
    performance.set_logger(lambda m, lvl="INFO": logs.append((lvl, m)))
    performance.set_gpu(False)
    performance.set_numba(True)

    orig = performance._eval_total_numba

    def bad_eval(x, peaks):
        y = orig(x, peaks)
        y[0] += 1e-6
        return y

    monkeypatch.setattr(performance, "_eval_total_numba", bad_eval)
    x = np.linspace(-1, 1, 20)
    peaks = [(1.0, 0.0, 0.5, 0.3)]
    performance.eval_total(x, peaks)
    assert logs and logs[0][0] == "WARN"
    assert performance.which_backend() == "numpy"
    performance.set_logger(None)

