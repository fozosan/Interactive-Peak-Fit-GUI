"""The Bayesian likelihood is evaluated for a whole half-ensemble in one call.

emcee's ``vectorize=True`` hands the log-probability a (walkers, dim) array.
The performance review measured the batched likelihood 7x faster than one call
per walker (727 to 104 us for 36 walkers, 82 % of a run being log_prob). The
likelihood keeps PR D's per-point sigma_i and PR E's joint baseline: the
weighted residual is projected off the baseline columns, and the scale term
counts n - m points.

The batch hands ``bayesian_ci`` a ``predict_batch``, the window model for rows
of parameter vectors at once; any other predictor is called once per row.
"""

import math

import numpy as np
import pytest

from batch import runner
from core.uncertainty import bayesian_ci
from tests.bayes_problem import model, mos2_problem, recording_sampler, sampled_index
from tests.synthetic_raman import BACKGROUND, TEMPLATE, WINDOW, curved_shape, mos2_spectrum, write_spectrum

emcee = pytest.importorskip("emcee")


def test_log_probability_is_vectorised_and_exact(monkeypatch):
    made = recording_sampler(monkeypatch)
    kwargs, info = mos2_problem(4)

    bayesian_ci(**kwargs, n_burn=2, n_steps=4, seed=1)

    sampler = made[-1]
    assert sampler.init_kwargs.get("vectorize") is True
    log_prob = getattr(sampler.log_prob_fn, "f", sampler.log_prob_fn)
    Z = sampler.start[:6].copy()
    lp = np.asarray(log_prob(Z), float)
    assert lp.shape == (6,)

    # By hand: the weighted residual projected off the baseline columns, Gaussian
    # with sd k sigma_i over n - m points, and k half-Cauchy (scale: the fit's RMS
    # weighted residual) with the Jacobian of sampling log k.
    th, y, sig = kwargs["theta_hat"], kwargs["y_all"], kwargs["sigma"]
    B = kwargs["fit_ctx"]["baseline_basis"]
    Q, _ = np.linalg.qr(B / sig[:, None])
    n_eff = y.size - B.shape[1]

    def proj(v):
        return v - Q @ (Q.T @ v)

    r0 = proj((y - model(th)) / sig)
    s = math.sqrt(float(r0 @ r0) / n_eff)
    idx = sampled_index(info["n_pk"])
    ref = []
    for z in Z:
        t = th.copy()
        t[idx] = z[:-1]
        k = math.exp(z[-1])
        r = proj((y - model(t)) / sig)
        ll = -0.5 * float(r @ r) / k**2 - n_eff * math.log(k) - float(np.sum(np.log(sig))) - 0.5 * n_eff * math.log(2 * math.pi)
        ref.append(ll + math.log(2.0 / math.pi) - math.log(s * (1.0 + (k / s) ** 2)) + z[-1])
    np.testing.assert_allclose(lp, ref, rtol=1e-10, atol=1e-8)

    # outside the bounds (a negative height) the probability is zero
    Z[0, 1] = -1.0
    assert np.isneginf(np.asarray(log_prob(Z[:2]), float)[0])


@pytest.mark.parametrize("baseline_fit", ["joint", "fixed"])
def test_batch_hands_the_bayesian_a_batched_predictor(baseline_fit, monkeypatch, tmp_path):
    captured = {}

    def capture(method, **kw):
        captured.update(kw)
        raise RuntimeError("captured")

    monkeypatch.setattr(runner, "route_uncertainty", capture)
    spec = tmp_path / "spec.csv"
    write_spectrum(spec, *mos2_spectrum(0, background=lambda x: BACKGROUND + curved_shape(x)))
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(tmp_path / "out"),
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "baseline_fit": baseline_fit,
        "bayes_walkers": 0,
        "bayes_burn": 10,
        "bayes_steps": 20,
        "bayes_thin": 1,
    }
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="bayesian")

    predict_rows = captured["fit_ctx"]["predict_batch"]
    predict = captured["model_eval"]
    th = np.asarray(captured["theta_hat"], float)
    rows = th + 1e-3 * np.abs(th) * np.random.default_rng(0).standard_normal((3, th.size))
    np.testing.assert_allclose(predict_rows(rows), np.stack([predict(t) for t in rows]), rtol=1e-12, atol=1e-9)
