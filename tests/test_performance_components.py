import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from infra import performance

pytestmark = pytest.mark.filterwarnings("ignore:.*")

x = np.linspace(-1, 1, 50)
peaks = [(1.0, -0.1, 0.4, 0.3), (0.5, 0.2, 0.2, 0.6)]


def _check_components():
    comps = performance.eval_components(x, peaks)
    total = performance.eval_total(x, peaks)
    A = performance.design_matrix(x, peaks)
    assert comps.shape == (len(peaks), x.size)
    assert A.shape == (x.size, len(peaks))
    assert np.allclose(comps.sum(axis=0), total, rtol=1e-10, atol=1e-12)
    assert np.allclose(A, comps.T, rtol=1e-10, atol=1e-12)


def test_components_numpy():
    performance.set_numba(False)
    performance.set_gpu(False)
    _check_components()


@pytest.mark.skipif(not getattr(performance, "_NUMBA_OK", False), reason="Numba not available")
def test_components_numba():
    performance.set_gpu(False)
    performance.set_numba(True)
    if performance.which_backend() != "numba":
        pytest.skip("Numba not usable")
    _check_components()
    performance.set_numba(False)


@pytest.mark.skipif(not getattr(performance, "_CUPY_OK", False), reason="CuPy not available")
def test_components_gpu():
    performance.set_numba(False)
    performance.set_gpu(True)
    if performance.which_backend() != "cupy":
        pytest.skip("CuPy not usable")
    _check_components()
    performance.set_gpu(False)

