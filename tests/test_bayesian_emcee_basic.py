import numpy as np, pytest
emcee = pytest.importorskip("emcee")
from core.uncertainty import bayesian_ci

def test_bayes_returns_stats_and_diag():
    x = np.linspace(-2, 2, 41)
    # true model: single PV approximated by gaussian-ish bump
    def model(th):
        c,h,w,eta = th
        return h*np.exp(-(x-c)**2/(2*w*w))
    th_true = np.array([0.0, 1.0, 0.6, 0.5])
    rng = np.random.default_rng(0)
    y = model(th_true) + rng.normal(0, 0.05, size=x.size)

    th0 = th_true + np.array([0.05, -0.1, 0.1, 0.0])
    res = bayesian_ci(th0, predict_full=lambda th: model(th), x_all=x, y_all=y,
                      locked_mask=np.array([False, False, False, True]),
                      n_burn=200, n_steps=600, thin=1, seed=123, return_band=True)
    assert "sigma" in res.stats
    assert res.diagnostics["n_draws"] > 0
    assert res.diagnostics["accept_frac_mean"] > 0
    assert res.label.startswith("Bayesian")
    if res.band is not None:
        x_b, lo, hi = res.band
        assert x_b.shape == lo.shape == hi.shape == x.shape
