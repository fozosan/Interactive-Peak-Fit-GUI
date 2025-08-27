import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from infra import performance

pytestmark = pytest.mark.filterwarnings("ignore:.*")


def _reference_components(x, peaks):
    comps = []
    for h, c, w, eta in peaks:
        w = max(w, 1e-12)
        eta = min(1.0, max(0.0, eta))
        dx = (x - c) / w
        ga = np.exp(-4.0 * np.log(2.0) * dx * dx)
        lo = 1.0 / (1.0 + 4.0 * dx * dx)
        comps.append(h * ((1.0 - eta) * ga + eta * lo))
    if comps:
        return np.vstack(comps)
    return np.zeros((0, x.size))


def _run_backend_check():
    x = np.linspace(-1, 1, 100)
    peaks = [(1.0, -0.2, 0.5, 0.2), (0.7, 0.3, 0.4, 0.6)]
    ref_comps = _reference_components(x, peaks)
    ref_total = ref_comps.sum(axis=0)

    y = performance.eval_total(x, peaks)
    comps = performance.eval_components(x, peaks)
    A = performance.design_matrix(x, peaks)

    assert y.dtype == np.float64
    assert comps.dtype == np.float64
    assert A.dtype == np.float64
    assert comps.shape == (len(peaks), x.size)
    assert A.shape == (x.size, len(peaks))

    assert np.allclose(y, ref_total, rtol=1e-10, atol=1e-12)
    assert np.allclose(comps, ref_comps, rtol=1e-10, atol=1e-12)
    assert np.allclose(A, ref_comps.T, rtol=1e-10, atol=1e-12)


def test_numpy_backend_matches_reference():
    performance.set_numba(False)
    performance.set_gpu(False)
    _run_backend_check()
    assert performance.which_backend() == "numpy"


@pytest.mark.skipif(not getattr(performance, "_NUMBA_OK", False), reason="Numba not available")
def test_numba_backend_agrees():
    performance.set_gpu(False)
    performance.set_numba(True)
    if performance.which_backend() != "numba":
        pytest.skip("Numba not usable")
    _run_backend_check()
    performance.set_numba(False)


@pytest.mark.skipif(not getattr(performance, "_CUPY_OK", False), reason="CuPy not available")
def test_gpu_backend_agrees():
    performance.set_numba(False)
    performance.set_gpu(True)
    if performance.which_backend() != "cupy":
        pytest.skip("CuPy not usable")
    _run_backend_check()
    performance.set_gpu(False)


@pytest.mark.skipif(
    performance.which_backend() == "cupy"
    or not getattr(performance, "_CUPY_OK", False),
    reason="CuPy not available",
)
def test_gpu_skipped():
    assert True

