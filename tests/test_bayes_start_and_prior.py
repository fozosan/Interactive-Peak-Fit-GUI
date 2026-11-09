"""Walker start and priors of the Bayesian (game plan 1.7, review finding 6, PR E).

- The walkers start in a tight ball around the fit, a tenth of each parameter's
  asymptotic SE, never in a box sized by the bounds. The batch's height bound is
  the solver's 1e12 sentinel, so the old ball, 1 % of a tenth of the bound
  width, put walkers about 1e9 counts out, and PR E measured Bayesian height
  intervals thousands of times the true sd in every case.
- The heights' upper bound is finite and data-driven: twice the larger of the
  data's range in the window and the fitted height. A peak's height cannot
  exceed the range by more than the noise unless the baseline dips under the
  data or the centre sits outside the window, and twice the range leaves room
  for both. A smaller bound from the caller (the GUI's 1.5 max(y)) is kept.
- The priors are flat inside the bounds. The old Gaussian prior centred on the
  fit (sd a tenth of the bound width) used the data twice and narrowed the
  posterior. The scale k is half-Cauchy as named: sampling log k, its density
  carries the Jacobian k, where the old code divided by k instead.
"""

import math

import numpy as np
import pytest

from core.uncertainty import bayesian_ci
from tests.bayes_problem import asymptotic_se, mos2_problem, recording_sampler, sampled_index

emcee = pytest.importorskip("emcee")


def test_walkers_start_in_a_tight_ball_scaled_by_the_asymptotic_se(monkeypatch):
    made = recording_sampler(monkeypatch)
    kwargs, info = mos2_problem(2)
    se = asymptotic_se(kwargs)
    idx = sampled_index(info["n_pk"])

    bayesian_ci(**kwargs, n_burn=10, n_steps=20, seed=4)

    start = made[-1].start
    assert start.shape == (made[-1].nwalkers, len(idx) + 1)
    z = (start[:, : len(idx)] - kwargs["theta_hat"][idx]) / se
    spread = z.std(axis=0)
    assert np.all(spread > 0.02) and np.all(spread < 0.3)  # 0.1 SE, give or take
    assert np.max(np.abs(z)) < 1.0
    # the scale k starts at the fit's RMS weighted residual, in the same tight ball
    log_k = start[:, -1]
    assert np.std(log_k) < 0.3 / math.sqrt(2.0 * (kwargs["y_all"].size - 4 - len(idx)))


def test_height_bound_is_finite_and_data_driven():
    kwargs, info = mos2_problem(3)
    y, th = kwargs["y_all"], kwargs["theta_hat"]
    n_pk = info["n_pk"]

    res = bayesian_ci(**kwargs, n_burn=300, n_steps=1200, seed=5)

    lo, hi = (np.asarray(b, float) for b in res.diagnostics["param_bounds"])
    cap = 2.0 * max(float(y.max() - y.min()), float(th[1 : 4 * n_pk : 4].max()))
    assert np.allclose(hi[1 : 4 * n_pk : 4], cap)
    # the posterior heights are the asymptotic ones, not intervals reaching the bound
    se = asymptotic_se(kwargs)
    idx = sampled_index(n_pk)
    rows = res.stats["rows"]
    for i in range(n_pk):
        h = rows[i]["height"]
        se_h = se[idx.index(4 * i + 1)]
        assert 0.75 < h["sd"] / se_h < 1.33
        assert h["p97_5"] < th[4 * i + 1] + 5.0 * se_h

    # A smaller bound from the caller (the GUI's kind, under twice the range) is kept.
    lo, hi = (np.array(b, float) for b in kwargs["bounds"])
    hi[1:8:4] = [80.0, 110.0]
    res = bayesian_ci(**dict(kwargs, bounds=(lo, hi)), n_burn=20, n_steps=40, seed=5)
    assert np.allclose(np.asarray(res.diagnostics["param_bounds"][1])[1:8:4], [80.0, 110.0])


def _height_only(seed=0, n=100):
    """One line whose height alone is sampled, and its exact posterior scale."""
    x = np.linspace(-10.0, 10.0, n)
    shape = 1.0 / (1.0 + (x / 1.5) ** 2)
    y = 20.0 * shape + np.random.default_rng(seed).standard_normal(n)
    h_hat = float(shape @ y / (shape @ shape))
    resid = y - h_hat * shape
    se = math.sqrt(float(resid @ resid) / (n - 1) / float(shape @ shape))
    return x, y, shape, h_hat, se


def test_prior_is_flat_inside_the_bounds():
    # The bounds sit 5 SE either side of the fit. A prior centred on the fit with
    # sd a tenth of their width (1 SE) would shrink the posterior sd to 0.7 SE.
    x, y, shape, h_hat, se = _height_only()

    def predict(th):
        return th[1] * shape

    res = bayesian_ci(
        np.array([0.0, h_hat, 3.0, 0.5]),
        predict_full=predict,
        x_all=x,
        y_all=y,
        bounds=(np.array([-1.0, h_hat - 5 * se, 0.1, 0.0]), np.array([1.0, h_hat + 5 * se, 10.0, 1.0])),
        locked_mask=np.array([True, False, True, True]),
        n_burn=500,
        n_steps=4000,
        seed=7,
        return_band=False,
    )
    # flat prior on h, about 1/k on k (half-Cauchy at its scale): Student-t, n - 1 dof
    n = y.size
    sd_exact = se * math.sqrt((n - 1) / (n - 3))
    assert 0.92 < res.stats["height"]["sd"][0] / sd_exact < 1.08
    assert abs(res.stats["height"]["est"][0] - h_hat) < 0.1 * se


def test_scale_prior_is_half_cauchy_with_its_jacobian():
    # Nothing but the noise scale is sampled; its posterior is known in closed form.
    n = 30
    x = np.linspace(0.0, 1.0, n)
    y = 2.0 * np.random.default_rng(3).standard_normal(n)
    rss = float(y @ y)
    s = math.sqrt(rss / n)  # the prior's scale: the fit's RMS residual

    res = bayesian_ci(
        np.array([0.0, 1.0, 1.0, 0.5]),
        predict_full=lambda th: np.zeros_like(x),
        x_all=x,
        y_all=y,
        locked_mask=np.ones(4, bool),
        n_walkers=16,
        n_burn=500,
        n_steps=6000,
        seed=11,
        return_band=False,
    )

    # p(u), u = log k: likelihood k^-n exp(-RSS / 2k^2), times HalfCauchy(k; s), times k
    u = np.linspace(math.log(s) - 1.5, math.log(s) + 1.5, 20001)
    k = np.exp(u)
    logp = -n * u - rss / (2.0 * k * k) - np.log1p((k / s) ** 2) + u
    p = np.exp(logp - logp.max())
    cdf = np.cumsum(p) / p.sum()
    median_exact = float(k[np.searchsorted(cdf, 0.5)])
    assert res.stats["sigma"]["est"] == pytest.approx(median_exact, rel=0.01)
