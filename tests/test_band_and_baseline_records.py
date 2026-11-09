"""Regressions 5 and 8 of the 2026-11-08 review of #253-#255, in the core path.

5. Compute band off suppressed only the bootstrap's band: the asymptotic method
   still made one, and the Bayesian one followed its own switch. Now no method
   returns a band unless one is wanted (pipeline.band_wanted).
8. With Apply baseline off, the TXT said "Baseline: fixed (als), held at its
   precomputed values", and _fit.csv said baseline_fit "fixed". Both now say
   there is no baseline ("none"), and so does batch's status table.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from batch import runner
from core import fit_api, pipeline
from core.peaks import Peak
from tests.conftest import bayes_knobs
from tests.test_batch_file_isolation import _config, _spectrum, _write


def _fit(tmp_path, use_baseline=True, **extra):
    x, y = _spectrum()
    settings = {**_config(tmp_path), "use_baseline": use_baseline, "perf_unc_workers": 1, **extra}
    seeds = [Peak(41.0, 90.0, 4.5, 0.5), Peak(59.0, 70.0, 5.5, 0.5)]
    mask = (x >= 20.0) & (x <= 80.0)
    base = np.full_like(x, 10.0) if use_baseline else np.zeros_like(x)
    res = fit_api.run_fit_consistent(x, y, seeds, settings, base, "add", mask,
                                     return_jacobian=True, return_predictors=True)
    assert res["fit_ok"]
    return x, y, mask, res, settings


def _band(raw):
    return raw.get("band") if isinstance(raw, dict) else getattr(raw, "band", None)


@pytest.mark.parametrize("method, extra", [
    ("asymptotic", {}),
    ("bootstrap", {"bootstrap_n": 30}),
    ("bayesian", {**bayes_knobs(walkers=16, burn=100, steps=300), "bayes_band_enabled": True,
                  "bayes_band_force": True}),
])
def test_compute_band_off_means_no_band_for_any_method(tmp_path, method, extra):
    x, y, mask, res, settings = _fit(tmp_path, **extra)
    on = pipeline.compute_uncertainty(res, x, y, mask, settings, method)
    off = pipeline.compute_uncertainty(res, x, y, mask, {**settings, "ui_band_pref_bootstrap": False}, method)
    assert _band(on) is not None
    assert _band(off) is None
    unc = pipeline.finish_uncertainty(off, res, settings, method)
    written = pipeline.write_uncertainty(tmp_path / "s", "s.csv", unc, res, settings)
    assert sorted(p.name for p in written) == ["s_uncertainty.csv", "s_uncertainty.txt"]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_with_apply_baseline_off_the_tables_say_there_is_no_baseline(tmp_path):
    x, y, mask, res, settings = _fit(tmp_path, use_baseline=False)
    raw = pipeline.compute_uncertainty(res, x, y, mask, settings, "asymptotic")
    unc = pipeline.finish_uncertainty(raw, res, settings, "asymptotic")
    pipeline.write_spectrum(tmp_path / "nb", "nb.csv", x=x, y_raw=y, fit=res, settings=settings, unc_norm=unc)
    txt = (tmp_path / "nb_uncertainty.txt").read_text(encoding="utf-8")
    assert "Baseline: none" in txt and "Baseline: fixed" not in txt
    assert {r["baseline_fit"] for r in _rows(tmp_path / "nb_fit.csv")} == {"none"}
    assert {r["baseline_fit"] for r in _rows(tmp_path / "nb_uncertainty.csv")} == {"none"}


def test_batch_with_use_baseline_off_says_none(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    _write(in_dir / "a.csv", *_spectrum())
    cfg = {**_config(out_dir), "use_baseline": False}
    runner.run_batch([str(in_dir / "a.csv")], cfg, compute_uncertainty=True, unc_method="Asymptotic")
    assert {r["baseline_fit"] for r in _rows(out_dir / "batch_files.csv")} == {"none"}
    assert {r["baseline_fit"] for r in _rows(out_dir / "a_fit.csv")} == {"none"}
    assert "Baseline: none" in (out_dir / "a_uncertainty.txt").read_text(encoding="utf-8")
