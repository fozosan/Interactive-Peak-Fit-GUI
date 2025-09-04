import numpy as np, pytest
emcee = pytest.importorskip("emcee")
from core.uncertainty import bayesian_ci

def _run(seed):
    x = np.linspace(0,1,21)
    def model(th): return th[1]*np.exp(-(x-th[0])**2/(2*th[2]**2))
    y = model(np.array([0.5,1.0,0.2])) + 0.05*np.random.default_rng(0).normal(size=x.size)
    th0 = np.array([0.48, 0.9, 0.22, 0.5])
    return bayesian_ci(th0, predict_full=model, x_all=x, y_all=y,
                       locked_mask=np.array([False,False,False,True]),
                       n_burn=200, n_steps=400, seed=seed, return_band=False)

def test_seed_repro():
    a = _run(7); b = _run(7)
    assert np.isclose(a.stats["p0"]["est"], b.stats["p0"]["est"], rtol=0, atol=1e-6)
