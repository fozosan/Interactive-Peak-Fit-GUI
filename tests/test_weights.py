import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.weights import robust_weights


def test_robust_weights_monotonic():
    r = np.linspace(0.0, 5.0, 50)
    for loss in ["soft_l1", "huber", "cauchy"]:
        w = robust_weights(loss, r, 1.0)
        assert np.all(w[:-1] >= w[1:])
    w_lin = robust_weights("linear", r, 1.0)
    assert w_lin is None
