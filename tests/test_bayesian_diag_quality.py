import numpy as np, pytest
emcee = pytest.importorskip("emcee")
from core.uncertainty import bayesian_ci

def test_diag_contains_ess_rhat():
    x = np.linspace(-1,1,31)
    def model(th): return th[1]*np.exp(-(x-th[0])**2/(2*th[2]**2))
    y = model(np.array([0.0,1.0,0.4])) + 0.05*np.random.default_rng(1).normal(size=x.size)
    th0 = np.array([0.02, 0.95, 0.45, 0.5])
    res = bayesian_ci(th0, predict_full=model, x_all=x, y_all=y,
                      locked_mask=np.array([False,False,False,True]),
                      n_burn=200, n_steps=400, seed=1, return_band=False)
    assert "ess_min" in res.diagnostics and "rhat_max" in res.diagnostics
