"""With a fit window, the batch uncertainty sees only the window.

The fit result held no window arrays, so the batch fell back to the whole
spectrum: the bootstrap failed on every file ("predict_full produced vector of
wrong size (got 1650, want 135)") and the Bayesian method sampled all 1650
points, with a noise sigma of 2.15 instead of about 1.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import uncertainty
from tests.synthetic_raman import BACKGROUND, TEMPLATE, WINDOW, mos2_spectrum, write_spectrum


def _config(out_dir, mode="add"):
    return {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "mode": mode,
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(out_dir),
        "output_base": "batch",
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "bootstrap_n": 20,
        "bayes_walkers": 0,
        "bayes_burn": 20,
        "bayes_steps": 60,
        "bayes_thin": 1,
        "perf_seed_all": True,
        "perf_seed": 3,
        "perf_unc_workers": 1,
    }


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def spectrum(tmp_path):
    path = tmp_path / "spec.csv"
    x, y = mos2_spectrum(21)
    write_spectrum(path, x, y)
    n_window = int(np.count_nonzero((x >= WINDOW[0]) & (x <= WINDOW[1])))
    return path, n_window


@pytest.mark.parametrize("mode", ["add", "subtract"])
def test_bootstrap_with_a_window_uses_the_window(tmp_path, spectrum, mode):
    path, n_window = spectrum
    out = tmp_path / "out"
    runner.run_batch([str(path)], _config(out, mode), compute_uncertainty=True, unc_method="bootstrap")
    status = _rows(out / "batch_files.csv")[0]
    assert status["unc_status"] == "ok", status["unc_message"]
    band = _rows(out / "spec_band.csv")
    assert len(band) == n_window
    xs = [float(r["x"]) for r in band]
    assert WINDOW[0] <= min(xs) and max(xs) <= WINDOW[1]
    # The band is of the curve the fit models: data with the baseline in add
    # mode, the baseline-subtracted signal in subtract mode.
    edge = float(band[0]["ci_lo"])
    if mode == "add":
        assert edge == pytest.approx(BACKGROUND, abs=3.0)
    else:
        assert abs(edge) < 3.0


def test_asymptotic_band_with_a_window_covers_the_window(tmp_path, spectrum):
    path, n_window = spectrum
    out = tmp_path / "out"
    runner.run_batch([str(path)], _config(out), compute_uncertainty=True, unc_method="asymptotic")
    assert _rows(out / "batch_files.csv")[0]["unc_status"] == "ok"
    assert len(_rows(out / "spec_uncertainty_band.csv")) == n_window


def test_bayesian_with_a_window_samples_the_window(tmp_path, spectrum, monkeypatch):
    path, n_window = spectrum
    seen = {}
    real = uncertainty.bayesian_ci

    def spy(*args, **kwargs):
        seen["n"] = int(np.asarray(kwargs["y_all"]).size)
        seen["n_pred"] = int(np.asarray(kwargs["predict_full"](kwargs["theta_hat"])).size)
        return real(*args, **kwargs)

    monkeypatch.setattr(uncertainty, "bayesian_ci", spy)
    out = tmp_path / "out"
    runner.run_batch([str(path)], _config(out), compute_uncertainty=True, unc_method="bayesian")
    status = _rows(out / "batch_files.csv")[0]
    assert status["unc_status"] == "ok", status["unc_message"]
    assert seen == {"n": n_window, "n_pred": n_window}
