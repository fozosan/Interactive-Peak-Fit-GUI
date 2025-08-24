import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.robust import irls_weights


def test_irls_weights_monotonic():
    r = np.linspace(0.0, 5.0, 50)
    for loss in ["soft_l1", "huber", "cauchy"]:
        w = irls_weights(r, loss, 1.0)
        assert np.all(w[:-1] >= w[1:])
    w_lin = irls_weights(r, "linear", 1.0)
    assert np.allclose(w_lin, 1.0)
