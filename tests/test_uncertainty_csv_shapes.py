import csv
import math
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io  # noqa: E402


def test_long_csv_has_both_ci_and_quantiles(tmp_path):
    unc_norm = {
        "label": "asymptotic",
        "stats": [{"center": {"est": 10.0, "sd": 2.0}}],
    }
    out = data_io.write_uncertainty_csv_legacy(tmp_path / "base", "sample.txt", unc_norm)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {"p2_5", "p97_5", "ci_lo", "ci_hi"} <= rows[0].keys()
    r = rows[0]
    expected_lo = 10.0 - data_io._Z * 2.0
    expected_hi = 10.0 + data_io._Z * 2.0
    assert float(r["p2_5"]) == pytest.approx(expected_lo)
    assert float(r["p97_5"]) == pytest.approx(expected_hi)
    assert float(r["ci_lo"]) == pytest.approx(expected_lo)
    assert float(r["ci_hi"]) == pytest.approx(expected_hi)


def test_sigma_to_fwhm_applied_before_export(tmp_path):
    raw = {"params": {"sigma": {"est": 1.0, "sd": 0.2}}}
    norm = data_io.normalize_unc_result(raw)
    fwhm = norm["stats"][0]["fwhm"]
    assert fwhm["est"] == pytest.approx(data_io._FWHM_SIGMA)
    out = data_io.write_uncertainty_csv_legacy(tmp_path / "base", "sample.txt", norm)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fwhm_rows = [r for r in rows if r["param"] == "fwhm"]
    assert fwhm_rows and float(fwhm_rows[0]["value"]) == pytest.approx(data_io._FWHM_SIGMA)
