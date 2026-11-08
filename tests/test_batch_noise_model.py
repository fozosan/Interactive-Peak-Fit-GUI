"""The batch fits with the per-spectrum noise model by default and records it.

Decision D1 (2026-11-08): var_i = s0^2 + g * S_i is estimated for each spectrum.
Batch uses it unless ``noise_model`` is ``"none"``. Its provenance (s0, g, n and
method) goes into the status table ``<output_base>_files.csv``, the uncertainty
CSVs and the uncertainty TXT. All three methods get the fit's sigma_i, and the
bootstrap refits are weighted by it, never re-estimated.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import fit_api
from core import uncertainty as U
from core.peaks import Peak
from core.uncertainty_router import route_uncertainty
from tests.conftest import bayes_knobs
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

NOISE_COLUMNS = ("noise_s0", "noise_g", "noise_n", "noise_method")
ALS = {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0}


def _config(out_dir, **extra):
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": dict(ALS),
        "output_dir": str(out_dir),
        "output_base": "batch",
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "perf_seed_all": True,
        "perf_seed": 7,
        "perf_unc_workers": 1,
    }
    cfg.update(extra)
    return cfg


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _spectrum(tmp_path, seed=41):
    path = tmp_path / "spec.csv"
    write_spectrum(path, *mos2_spectrum(seed))
    return path


def test_batch_estimates_the_noise_model_by_default_and_records_it(tmp_path):
    spec = _spectrum(tmp_path)
    out = tmp_path / "out"
    runner.run_batch([str(spec)], _config(out), compute_uncertainty=True, unc_method="asymptotic")

    header, (status,) = _rows(out / "batch_files.csv")
    for col in NOISE_COLUMNS:
        assert col in header
    assert status["status"] == "ok" and status["unc_status"] == "ok", status
    assert status["noise_method"] == "fit_residuals_d4"
    x, _y = mos2_spectrum(41)
    assert int(status["noise_n"]) == int(np.count_nonzero((x >= WINDOW[0]) & (x <= WINDOW[1])))
    # The detector's floor is 0.91 counts here. The app's ALS baseline bends
    # into the lines and its misfit there reads as signal-dependent noise, so g
    # comes out above the detector's 0.0451 until the baseline is fitted with
    # the peaks (game plan 1.6).
    s0, g = float(status["noise_s0"]), float(status["noise_g"])
    assert 0.6 < s0 < 1.3 and 0.0 < g < 0.5

    header, rows = _rows(out / "spec_uncertainty.csv")
    for col in NOISE_COLUMNS:
        assert col in header
    for r in rows:
        assert [r[c] for c in NOISE_COLUMNS] == [status[c] for c in NOISE_COLUMNS]

    header, rows = _rows(out / "batch_uncertainty_long.csv")
    for col in NOISE_COLUMNS:
        assert col in header
    assert all(r["noise_method"] == "fit_residuals_d4" for r in rows)

    txt = (out / "spec_uncertainty.txt").read_text(encoding="utf-8")
    assert "Noise model: var = s0^2 + g*S" in txt
    assert f"s0 = {s0:.6g}" in txt and f"g = {g:.6g}" in txt


def test_noise_model_none_fits_unweighted_and_leaves_the_columns_blank(tmp_path):
    spec = _spectrum(tmp_path)
    out = tmp_path / "out"
    runner.run_batch(
        [str(spec)], _config(out, noise_model="none"), compute_uncertainty=True, unc_method="asymptotic"
    )
    _h, (status,) = _rows(out / "batch_files.csv")
    assert [status[c] for c in NOISE_COLUMNS] == ["", "", "", ""]
    _h, rows = _rows(out / "spec_uncertainty.csv")
    assert all(r["noise_method"] == "" for r in rows)
    assert "Noise model" not in (out / "spec_uncertainty.txt").read_text(encoding="utf-8")

    x, y = mos2_spectrum(41)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    single = fit_api.run_fit_consistent(
        x, y, seeds, {"solver": "modern_vp", "baseline": dict(ALS)}, None, "add", mask
    )
    _h, fit_rows = _rows(out / "spec_fit.csv")
    batch_theta = [float(r[k]) for r in fit_rows for k in ("center", "height", "fwhm", "eta")]
    np.testing.assert_allclose(batch_theta, single["theta"], rtol=1e-10)


def test_batch_matches_the_core_weighted_fit_and_its_asymptotic_se(tmp_path):
    """Single-file and batch parity with the noise model on: the same fit, the
    same sigma_i, the same weighted standard errors."""
    spec = _spectrum(tmp_path, 42)
    out = tmp_path / "out"
    runner.run_batch([str(spec)], _config(out), compute_uncertainty=True, unc_method="asymptotic")
    _h, rows = _rows(out / "spec_uncertainty.csv")
    se_batch = {(int(r["peak"]), r["param"]): float(r["stderr"]) for r in rows if r["param"] != "eta"}
    _h, fit_rows = _rows(out / "spec_fit.csv")
    assert all(r["solver_weight"] == "noise_model" for r in fit_rows)
    batch_theta = [float(r[k]) for r in fit_rows for k in ("center", "height", "fwhm", "eta")]

    x, y = mos2_spectrum(42)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    cfg = {"solver": "modern_vp", "baseline": dict(ALS), "noise_model": "per_spectrum"}
    res = fit_api.run_fit_consistent(
        x, y, seeds, cfg, None, "add", mask, return_jacobian=True, return_predictors=True
    )
    np.testing.assert_allclose(batch_theta, res["theta"], rtol=1e-10)
    xf = x[mask]
    unc = route_uncertainty(
        "asymptotic", theta_hat=res["theta"], residual_fn=res["residual_fn"], jacobian=res["jacobian"],
        model_eval=res["predict_fit"], x_all=xf, y_all=y[mask],
        fit_ctx={"locked_mask": res["locked_mask"], "bounds": res["bounds"], "x_all": xf,
                 "noise_sigma": res["noise_sigma"]},
    )
    for (peak, param), se in se_batch.items():
        j = 4 * (peak - 1) + ("center", "height", "fwhm").index(param)
        assert se == pytest.approx(unc.stats[f"p{j}"]["sd"], rel=1e-9), (peak, param)


def test_batch_bootstrap_refits_are_weighted_by_the_fit_sigma(tmp_path, monkeypatch):
    calls = []
    real = fit_api.run_fit_consistent

    def spy(x, y, peaks_in, cfg, *args, **kwargs):
        out = real(x, y, peaks_in, cfg, *args, **kwargs)
        calls.append((dict(cfg), out))
        return out

    monkeypatch.setattr(fit_api, "run_fit_consistent", spy)
    spec = _spectrum(tmp_path, 43)
    out = tmp_path / "out"
    cfg = _config(out, bootstrap_n=6, unc_boot_solver="modern_vp")
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    _h, (status,) = _rows(out / "batch_files.csv")
    assert status["unc_status"] == "ok", status

    (first_cfg, first), *refits = calls
    assert first_cfg.get("noise_model") == "per_spectrum"
    sigma = first["noise_sigma"]
    assert sigma is not None and len(refits) == 6
    for cfg_r, out_r in refits:
        np.testing.assert_array_equal(np.asarray(cfg_r["noise_sigma"]), sigma)
        assert out_r["noise_model"] is None  # fixed weights: nothing re-estimated
        np.testing.assert_array_equal(out_r["noise_sigma"], sigma)


def test_batch_bayesian_gets_the_fit_sigma(tmp_path, monkeypatch):
    seen = {}
    real = U.bayesian_ci

    def spy(*args, **kwargs):
        seen["sigma"] = kwargs.get("sigma")
        return real(*args, **kwargs)

    monkeypatch.setattr(U, "bayesian_ci", spy)
    spec = _spectrum(tmp_path, 44)
    out = tmp_path / "out"
    cfg = _config(out, **bayes_knobs(burn=50, steps=150))
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="bayesian")
    _h, (status,) = _rows(out / "batch_files.csv")
    assert status["unc_status"] == "ok", status
    x, _y = mos2_spectrum(44)
    n_win = int(np.count_nonzero((x >= WINDOW[0]) & (x <= WINDOW[1])))
    assert seen["sigma"] is not None and np.asarray(seen["sigma"]).shape == (n_win,)
    assert float(np.min(seen["sigma"])) == pytest.approx(float(status["noise_s0"]), rel=0.02)
