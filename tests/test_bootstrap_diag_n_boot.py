import numpy as np
from core.uncertainty import bootstrap_ci

def test_diag_reports_n_boot():
    th = np.array([1.0,1.0,1.0,0.5])
    r = np.array([0.0, 1.0, 0.0])
    J = np.eye(4)
    x = np.array([0.0,1.0,2.0]); y = np.array([0.0,1.0,0.0])
    res = bootstrap_ci(
        th,
        r,
        J,
        x_all=x,
        y_all=y,
        fit_ctx={"refit": lambda ti, lm, b, x, y: ti},
        n_boot=10,
        seed=0,
    )
    assert res.diagnostics.get("n_boot") == 10
