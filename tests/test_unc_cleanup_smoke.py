import numpy as np
import importlib


def test_helpers_exist_and_basic_behave():
    u = importlib.import_module("core.uncertainty")
    for name in (
        "_norm_solver_and_sharing",
        "_build_residual_vector",
        "_relabel_by_center",
        "_validate_vector_length",
    ):
        assert hasattr(u, name), f"missing helper {name}"

    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([0.5, 2.5, 2.5, 5.0])
    resid = y - yhat
    r = u._build_residual_vector(residual=resid, y_all=y, y_hat=yhat, mode="raw", center=False)
    assert np.allclose(r, resid)

    ref = np.array([10, 1, 2, 0.3, 50, 2, 3, 0.7])
    new = np.array([50, 2, 3, 0.7, 10, 1, 2, 0.3])
    reordered = u._relabel_by_center(new, ref, block=4)
    assert np.allclose(reordered, ref)
