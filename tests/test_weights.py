import numpy as np
import pathlib, sys, warnings

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.weights import robust_weights


def test_robust_weights_monotonic():
    r = np.linspace(0.0, 5.0, 50)
    for loss in ["soft_l1", "huber", "cauchy"]:
        w = robust_weights(r, loss, 1.0)
        assert np.all(w[:-1] >= w[1:])
    w_lin = robust_weights(r, "linear", 1.0)
    assert w_lin is None


def test_robust_weights_no_runtime_warning():
    r = np.array([0.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        robust_weights(r, "huber", 1.0)
        assert not any(issubclass(v.category, RuntimeWarning) for v in w)
