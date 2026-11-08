"""The uncertainty writers write what was computed, at the level asked for.

- A missing statistic is a blank cell. It was written as 0.0, so a NaN
  standard error became a zero-width interval, and a dof <= 0 became 1.
- The confidence level follows alpha in the quantile columns, the TXT and the
  band header. Labels said 95 % whatever alpha was, and a CI rebuilt from the
  standard error always used 1.96.
- An interval that reaches a parameter bound is flagged.
"""

import csv
import math

import numpy as np
import pytest

from batch import runner
from core import data_io
from core.uncertainty import asymptotic_ci
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

Z90 = 1.6448536269514722


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _missing_result():
    nan = float("nan")
    return {
        "label": "Asymptotic",
        "rmse": nan,
        "dof": 0,
        "stats": [
            {
                "index": 1,
                "center": {"est": 383.0, "sd": nan},
                "height": {"est": 70.0, "sd": 1.5, "p2_5": nan, "p97_5": nan},
                "fwhm": {"est": 2.5, "sd": nan, "p2_5": 2.2, "p97_5": 2.9},
                "eta": {"est": 0.5, "sd": nan},
            }
        ],
    }


def test_missing_statistics_are_written_blank(tmp_path):
    long_csv, wide_csv = data_io.write_uncertainty_csvs(
        tmp_path / "probe.csv", "probe.csv", data_io.normalize_unc_result(_missing_result()), write_wide=True
    )
    _, rows = _rows(long_csv)
    by = {r["param"]: r for r in rows}
    for row in rows:
        assert row["rmse"] == "" and row["dof"] == ""
        assert "nan" not in ",".join(row.values()).lower()
    assert by["center"]["stderr"] == by["center"]["ci_lo"] == by["center"]["p2_5"] == ""
    assert float(by["center"]["value"]) == 383.0
    assert by["eta"]["stderr"] == by["eta"]["ci_hi"] == ""
    # height: the CI is rebuilt from the standard error
    assert float(by["height"]["p2_5"]) == pytest.approx(70.0 - 1.959963984540054 * 1.5)
    # fwhm: the given interval survives a missing standard error
    assert by["fwhm"]["stderr"] == "" and float(by["fwhm"]["p97_5"]) == 2.9

    _, wide = _rows(wide_csv)
    assert wide[0]["center_stderr"] == wide[0]["center_ci_lo"] == wide[0]["dof"] == ""


def test_batch_long_writer_writes_missing_values_blank(tmp_path):
    rows = list(data_io.iter_uncertainty_rows("a.csv", data_io.normalize_unc_result(_missing_result())))
    long_path, legacy_path = data_io.write_batch_uncertainty_long(tmp_path, rows)
    for path in (long_path, legacy_path):
        _, out = _rows(path)
        by = {r["param"]: r for r in out}
        assert by["center"]["stderr"] == by["center"]["p2_5"] == by["center"]["rmse"] == ""
        assert by["center"]["dof"] == ""
        assert float(by["height"]["stderr"]) == 1.5


def test_alpha_sets_the_quantile_columns_the_txt_and_rebuilt_intervals(tmp_path):
    unc = {
        "label": "Asymptotic",
        "alpha": 0.10,
        "rmse": 1.0,
        "dof": 129,
        "stats": [{"index": 1, "center": {"est": 383.0, "sd": 0.1}, "height": {"est": 70.0, "sd": 2.0},
                   "fwhm": {"est": 2.5, "sd": 0.1}, "eta": {"est": 0.5, "sd": float("nan")}}],
    }
    norm = data_io.normalize_unc_result(unc)
    long_csv, _ = data_io.write_uncertainty_csvs(tmp_path / "s.csv", "s.csv", norm, write_wide=False)
    header, rows = _rows(long_csv)
    assert "p5" in header and "p95" in header and "p2_5" not in header
    centre = next(r for r in rows if r["param"] == "center")
    assert float(centre["p5"]) == pytest.approx(383.0 - Z90 * 0.1)
    assert float(centre["ci_hi"]) == pytest.approx(383.0 + Z90 * 0.1)

    txt = tmp_path / "s_uncertainty.txt"
    data_io.write_uncertainty_txt(txt, unc, file_path="s.csv")
    text = txt.read_text(encoding="utf-8")
    assert "90% CI" in text and "95%" not in text

    out = list(data_io.iter_uncertainty_rows("s.csv", norm))
    batch_long, _ = data_io.write_batch_uncertainty_long(tmp_path, out)
    header, _ = _rows(batch_long)
    assert "p5" in header and "p95" in header


def test_interval_reaching_a_bound_is_flagged(tmp_path):
    theta = np.array([383.0, 0.5, 3.0, 0.5])
    J = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]] * 4)
    r = np.array([1.0, -1.0, 0.5] * 4)
    lo = np.array([340.0, 0.0, 1.5, 0.0])
    hi = np.array([440.0, 1e12, 50.0, 1.0])
    res = asymptotic_ci(theta, r, J, lambda th: np.zeros(12), locked_mask=[False, False, False, True],
                        bounds=(lo, hi), xgrid=np.arange(12.0))
    # rss 9 over dof 12 - 3 gives sd 0.5 each: height 0.5 +- 0.98 crosses its
    # lower bound 0, centre and FWHM stay inside theirs, eta is locked
    assert res.stats["p1"]["sd"] == pytest.approx(0.5)
    assert res.stats["p1"]["p2.5"] == 0.0
    long_csv, _ = data_io.write_uncertainty_csvs(
        tmp_path / "f.csv", "f.csv", data_io.normalize_unc_result(res), write_wide=False
    )
    header, rows = _rows(long_csv)
    assert "flags" in header
    flags = {r["param"]: r["flags"] for r in rows}
    assert flags == {"center": "", "height": "ci_at_bound", "fwhm": "", "eta": ""}


def test_batch_band_header_and_status_follow_alpha_and_flags(tmp_path):
    spectrum = tmp_path / "spec.csv"
    write_spectrum(spectrum, *mos2_spectrum(31))
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(tmp_path / "out"),
        "output_base": "batch",
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "unc_alpha": 0.10,
    }
    runner.run_batch([str(spectrum)], cfg, compute_uncertainty=True, unc_method="asymptotic")
    header, _ = _rows(tmp_path / "out" / "spec_uncertainty_band.csv")
    assert header == ["x", "y_lo90", "y_hi90"]
    header, rows = _rows(tmp_path / "out" / "spec_uncertainty.csv")
    assert "p5" in header and "p95" in header
    centre = next(r for r in rows if r["param"] == "center" and r["peak"] == "1")
    width = float(centre["ci_hi"]) - float(centre["ci_lo"])
    assert width == pytest.approx(2 * Z90 * float(centre["stderr"]), rel=1e-9)
    text = (tmp_path / "out" / "spec_uncertainty.txt").read_text(encoding="utf-8")
    assert "90% CI" in text and "95%" not in text
    status_header, status = _rows(tmp_path / "out" / "batch_files.csv")
    assert "flags" in status_header
    assert status[0]["flags"] == ""
