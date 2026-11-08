"""A bootstrap draw counts only when its refit converged.

Before, a draw counted whenever its theta was finite: the solver's own flag was
ignored, a dict result with ``fit_ok`` False still counted when strict refits
were off, and the batch turned an exception into a successful draw at the
start point. The estimate was the mean of the draws. Now a refit that fails or
does not converge is a failure, the estimate is the fit, and the standard error
is the spread of the draws about the fit.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import fit_api
from core.peaks import Peak
from core.uncertainty import bootstrap_ci
from tests.synthetic_raman import TEMPLATE, mos2_spectrum, write_spectrum


def _toy(n=40):
    x = np.linspace(0.0, 1.0, n)
    theta = np.array([0.5, 1.0, 0.2, 0.5])
    y_hat = np.exp(-((x - 0.5) / 0.1) ** 2)
    residual = np.where(np.arange(n) % 2 == 0, 0.1, -0.1)
    return x, theta, y_hat, residual


def test_non_converged_core_refits_are_failures(monkeypatch):
    x, theta, y_hat, residual = _toy()
    calls = {"failed": 0}

    def fake_fit(x, y, peaks_in, cfg, baseline=None, mode="add", fit_mask=None,
                 reheight=False, rng_seed=None, verbose=False,
                 return_jacobian=False, return_predictors=False):
        start = np.array([v for p in peaks_in for v in (p.center, p.height, p.fwhm, p.eta)], float)
        if y[0] < y_hat[0]:  # about half the draws: the solver stops far away, unconverged
            calls["failed"] += 1
            return {"theta": start + np.array([50.0, 0.0, 0.0, 0.0]), "fit_ok": False}
        return {"theta": start, "fit_ok": True}

    monkeypatch.setattr(fit_api, "run_fit_consistent", fake_fit)
    res = bootstrap_ci(
        theta=theta,
        residual=residual,
        jacobian=np.ones((x.size, theta.size)),
        predict_full=lambda th: y_hat,
        x_all=x,
        y_all=y_hat + residual,
        fit_ctx={"peaks": [Peak(0.5, 1.0, 0.2, 0.5)], "solver": "modern_vp", "strict_refit": True},
        n_boot=24,
        seed=1,
        return_band=False,
    )
    d = res.diagnostics
    assert calls["failed"] > 0
    assert 0 < d["n_fail"] < 24 and d["n_success"] + d["n_fail"] == 24
    # The unconverged refits 50 units away never reach the statistics.
    assert res.stats["p0"]["sd"] == pytest.approx(0.0, abs=1e-12)


def test_unconverged_refit_is_a_failure_even_without_strict_refits():
    x, theta, y_hat, residual = _toy()

    def refit(theta_init, locked_mask, bounds, x_in, y_in):
        return {"theta": np.asarray(theta_init, float) + 10.0, "fit_ok": False}

    with pytest.raises(RuntimeError, match="Insufficient successful bootstrap refits"):
        bootstrap_ci(
            theta=theta,
            residual=residual,
            jacobian=np.ones((x.size, theta.size)),
            predict_full=lambda th: y_hat,
            x_all=x,
            y_all=y_hat + residual,
            fit_ctx={"refit": refit, "peaks": [object()], "strict_refit": False,
                     "allow_linear_fallback": False},
            n_boot=8,
            seed=0,
            return_band=False,
        )


def test_estimate_is_the_fit_and_se_is_the_spread_about_it():
    x, theta, y_hat, residual = _toy()
    offset = np.array([0.3, 0.0, 0.0, 0.0])

    def biased_refit(theta_init, locked_mask, bounds, x_in, y_in):
        return np.asarray(theta_init, float) + offset, True

    res = bootstrap_ci(
        theta=theta,
        residual=residual,
        jacobian=np.ones((x.size, theta.size)),
        predict_full=lambda th: y_hat,
        x_all=x,
        y_all=y_hat + residual,
        fit_ctx={"refit": biased_refit, "peaks": [object()], "strict_refit": True},
        n_boot=10,
        seed=0,
        return_band=False,
    )
    centre = res.stats["p0"]
    # Every draw is the fit + 0.3: the estimate stays the fit, and the spread
    # about the fit is 0.3 (about the draws' own mean it would be 0).
    assert centre["est"] == theta[0]
    assert centre["sd"] == pytest.approx(0.3, rel=1e-12)
    assert centre["p2.5"] == pytest.approx(0.8, rel=1e-12)


def test_unseeded_bootstrap_is_reproducible():
    x, theta, y_hat, residual = _toy()
    residual = np.linspace(-0.2, 0.2, x.size)

    def echo_shift(theta_init, locked_mask, bounds, x_in, y_in):
        th = np.asarray(theta_init, float).copy()
        th[1] += float(np.mean(y_in - y_hat))  # depends on the resampled residuals
        return th, True

    runs = [
        bootstrap_ci(
            theta=theta,
            residual=residual,
            jacobian=np.ones((x.size, theta.size)),
            predict_full=lambda th: y_hat,
            x_all=x,
            y_all=y_hat + residual,
            fit_ctx={"refit": echo_shift, "peaks": [object()], "strict_refit": True},
            n_boot=12,
            seed=None,
            return_band=False,
        )
        for _ in range(2)
    ]
    assert runs[0].stats["p1"]["sd"] > 0
    assert runs[0].stats["p1"]["sd"] == runs[1].stats["p1"]["sd"]


def test_batch_refit_error_is_a_failure_not_a_draw_at_the_start(tmp_path, monkeypatch):
    spectrum = tmp_path / "spec.csv"
    write_spectrum(spectrum, *mos2_spectrum(3))
    real_fit = fit_api.run_fit_consistent
    calls = {"n": 0}

    def fit_then_fail(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the file's own fit
            return real_fit(*args, **kwargs)
        raise RuntimeError("refit exploded")

    monkeypatch.setattr(fit_api, "run_fit_consistent", fit_then_fail)
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(tmp_path / "out"),
        "output_base": "batch",
        "bootstrap_n": 8,
        "perf_unc_workers": 1,
    }
    runner.run_batch([str(spectrum)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    with open(tmp_path / "out" / "batch_files.csv", newline="", encoding="utf-8") as fh:
        status = next(csv.DictReader(fh))
    assert status["status"] == "ok"
    assert status["unc_status"] == "failed"
    assert "Insufficient successful bootstrap refits" in status["unc_message"]
    assert not (tmp_path / "out" / "spec_uncertainty.csv").exists()
