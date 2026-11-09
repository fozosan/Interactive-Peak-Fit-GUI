"""Bayesian diagnostics from the integrated autocorrelation time (game plan 1.7).

The walkers of one affine-invariant ensemble move together and start in one
ball, so they are not independent chains. emcee's own measure is the integrated
autocorrelation time tau (``get_autocorr_time``, Goodman and Weare 2010); the
ESS is N / tau for the N draws kept, and a chain under 50 tau is too short even
to trust tau (emcee's rule). tau and the short-chain flag are computed for every
run; ess_min, rhat_max and mcse_mean, which gate the band, only with
diagnostics on, as before. rhat_max, R-hat with each walker taken as a chain,
is kept for display and documented as that heuristic.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core.uncertainty import bayesian_ci
from tests.bayes_problem import recording_sampler
from tests.synthetic_raman import (
    SI_BACKGROUND, SI_CURVE_SCALE, SI_LINES, SI_TEMPLATE, SI_WINDOW, curved_shape, mos2_spectrum, write_spectrum,
)

emcee = pytest.importorskip("emcee")


def _line(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = 1.3 + 0.4 * x + rng.normal(scale=0.03, size=n)
    X = np.column_stack([np.ones_like(x), x])
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    return x, y, np.asarray(th, float), (lambda t: t[0] + t[1] * x)


def test_diagnostics_come_from_the_integrated_autocorrelation_time(monkeypatch):
    made = recording_sampler(monkeypatch)
    x, y, th, model = _line()

    res = bayesian_ci(
        th, predict_full=model, x_all=x, y_all=y, fit_ctx={"bayes_diagnostics": True},
        n_walkers=16, n_burn=300, n_steps=3000, thin=2, seed=3, return_band=False,
    )

    d = res.diagnostics
    tau = made[-1].get_autocorr_time(discard=300, thin=2, tol=0)
    np.testing.assert_allclose(d["tau"], tau, rtol=1e-12)
    assert d["tau_max"] == pytest.approx(float(np.max(tau)))
    n_draws = (3000 // 2) * 16
    assert d["ess_min"] == pytest.approx(n_draws / (float(np.max(tau)) / 2))  # N / tau, tau in draws
    assert d["chain_tau_ratio"] == pytest.approx(3000 / float(np.max(tau)))
    assert d["chain_short"] is False
    assert d["rhat_method"] == "split_walkers"


def test_a_short_chain_is_flagged_even_with_diagnostics_off():
    x, y, th, model = _line()

    res = bayesian_ci(
        th, predict_full=model, x_all=x, y_all=y, fit_ctx={"bayes_diagnostics": False},
        n_walkers=16, n_burn=20, n_steps=60, seed=3, return_band=False,
    )

    d = res.diagnostics
    assert d["chain_short"] is True
    assert d["chain_tau_ratio"] < 50
    assert any("autocorrelation time" in note for note in d.get("notes", []))
    assert np.isnan(d["ess_min"])  # the band-gating metrics stay behind the switch


def _si_batch(tmp_path, name, burn, steps):
    folder = tmp_path / name
    folder.mkdir()
    spec = folder / "si.csv"
    write_spectrum(spec, *mos2_spectrum(1, SI_LINES, background=lambda x: SI_BACKGROUND + curved_shape(x, SI_WINDOW, SI_CURVE_SCALE)))
    cfg = {
        "peaks": [dict(p) for p in SI_TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(folder / "out"),
        "fit_xmin": SI_WINDOW[0],
        "fit_xmax": SI_WINDOW[1],
        "perf_seed_all": True,
        "perf_seed": 21,
        "bayes_walkers": 0,
        "bayes_burn": burn,
        "bayes_steps": steps,
        "bayes_thin": 1,
    }
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="bayesian")
    with open(folder / "out" / "batch_files.csv", newline="", encoding="utf-8") as fh:
        return next(csv.DictReader(fh))


def test_batch_flags_a_short_chain(tmp_path):
    short = _si_batch(tmp_path, "short", 20, 60)
    assert short["unc_status"] == "ok"
    assert "mcmc_short_chain" in short["flags"].split(";")
    assert "autocorrelation time" in short["unc_message"]

    long = _si_batch(tmp_path, "long", 300, 5000)
    assert long["unc_status"] == "ok"
    assert "mcmc_short_chain" not in long["flags"].split(";")
