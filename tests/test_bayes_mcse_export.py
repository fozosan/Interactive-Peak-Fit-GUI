import csv
import math
from pathlib import Path

import numpy as np
import pytest

from core.uncertainty import _mcse_quantile
from core import data_io


def _fake_bayes_result(n_draws: int = 2000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(n_draws, 4))
    a_lo = alpha / 2.0
    a_hi = 1.0 - a_lo
    param_stats = {}
    stats_rows = []
    names = ("center", "height", "fwhm", "eta")
    row_entry = {}
    for idx, name in enumerate(names):
        sample = draws[:, idx]
        est = float(np.mean(sample))
        sd = float(np.std(sample, ddof=1))
        q_lo = float(np.quantile(sample, a_lo))
        q_hi = float(np.quantile(sample, a_hi))
        lo_mcse = float(_mcse_quantile(sample, a_lo))
        hi_mcse = float(_mcse_quantile(sample, a_hi))
        param_stats[name] = {
            "est": [est],
            "sd": [sd],
            "p2_5": [q_lo],
            "p97_5": [q_hi],
            "ci_lo_mcse": [lo_mcse],
            "ci_hi_mcse": [hi_mcse],
        }
        row_entry[name] = {
            "est": est,
            "sd": sd,
            "ci_lo": q_lo,
            "ci_hi": q_hi,
            "p2_5": q_lo,
            "p97_5": q_hi,
            "ci_lo_mcse": lo_mcse,
            "ci_hi_mcse": hi_mcse,
        }
    stats_rows.append(row_entry)
    return {
        "label": "Bayesian (MCMC)",
        "method": "bayesian",
        "rmse": 0.1,
        "dof": 50,
        "param_stats": param_stats,
        "stats": stats_rows,
    }


def _fake_asymp_result():
    stats_rows = []
    param_stats = {}
    names = ("center", "height", "fwhm", "eta")
    for name in names:
        block = {
            "est": [0.0],
            "sd": [1.0],
            "p2_5": [-1.0],
            "p97_5": [1.0],
        }
        param_stats[name] = block
    stats_rows.append(
        {
            name: {
                "est": 0.0,
                "sd": 1.0,
                "ci_lo": -1.0,
                "ci_hi": 1.0,
                "p2_5": -1.0,
                "p97_5": 1.0,
            }
            for name in names
        }
    )
    return {
        "label": "Asymptotic (JᵀJ)",
        "method": "asymptotic",
        "rmse": 0.2,
        "dof": 40,
        "param_stats": param_stats,
        "stats": stats_rows,
    }


def _read_header(path: Path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        return next(reader)


@pytest.mark.parametrize("q", [0.025, 0.5, 0.975])
def test_mcse_quantile_reasonable(q):
    rng = np.random.default_rng(123)
    samples = rng.standard_normal(5000)
    val = _mcse_quantile(samples, q)
    assert math.isfinite(val)
    assert val >= 0.0


def test_mcse_quantile_insufficient():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(10)
    val = _mcse_quantile(samples, 0.5, batches=50)
    assert math.isnan(val)


def test_single_export_includes_mcse_columns(tmp_path: Path):
    res = _fake_bayes_result()
    base = tmp_path / "sample"
    _, wide_path = data_io.write_uncertainty_csvs(base, "input.dat", res, write_wide=True)
    wide_csv = Path(wide_path)
    header = _read_header(wide_csv)
    assert "center_ci_lo_mcse" in header and "center_ci_hi_mcse" in header
    assert header.index("center_ci_lo_mcse") > header.index("center_ci_hi")
    assert header.index("center_ci_hi_mcse") > header.index("center_ci_lo_mcse")


def test_non_bayes_blanks_in_mcse_columns(tmp_path: Path):
    res = _fake_asymp_result()
    base = tmp_path / "sample"
    _, wide_path = data_io.write_uncertainty_csvs(base, "input.dat", res, write_wide=True)
    wide_csv = Path(wide_path)
    with wide_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for row in rows:
        assert row.get("center_ci_lo_mcse", "") in ("", None)
        assert row.get("center_ci_hi_mcse", "") in ("", None)
        assert row.get("eta_ci_lo_mcse", "") in ("", None)


def _batch_rows_from_result(file_id: str, res: dict):
    stats_source = res.get("param_stats") or {}
    norm_rows, _ = data_io._normalize_param_stats(stats_source)
    rows = []
    for idx, rec in enumerate(norm_rows, start=1):
        row = {
            "file_id": file_id,
            "peak": idx,
            "method": res.get("label"),
            "rmse": res.get("rmse"),
            "dof": res.get("dof"),
        }
        for name in ("center", "height", "fwhm", "eta"):
            row[name] = rec.get(f"{name}_est")
            row[f"{name}_stderr"] = rec.get(f"{name}_sd")
            row[f"{name}_ci_lo"] = rec.get(f"{name}_p2_5")
            row[f"{name}_ci_hi"] = rec.get(f"{name}_p97_5")
            row[f"{name}_ci_lo_mcse"] = rec.get(f"{name}_ci_lo_mcse")
            row[f"{name}_ci_hi_mcse"] = rec.get(f"{name}_ci_hi_mcse")
        rows.append(row)
    return rows


def test_batch_export_parity(tmp_path: Path):
    bayes_res = _fake_bayes_result()
    asymp_res = _fake_asymp_result()
    rows = []
    rows.extend(_batch_rows_from_result("fileA", bayes_res))
    rows.extend(_batch_rows_from_result("fileB", asymp_res))
    out_csv = tmp_path / "batch_uncertainty_wide.csv"
    data_io.write_batch_uncertainty_csv(rows, out_csv)
    header = _read_header(out_csv)
    assert "center_ci_lo_mcse" in header and "center_ci_hi_mcse" in header
    assert header.index("center_ci_lo_mcse") > header.index("center_ci_hi")
    with out_csv.open(newline="") as fh:
        data_rows = list(csv.DictReader(fh))
    assert any(row.get("file_id") == "fileA" and row.get("center_ci_lo_mcse") not in ("", None) for row in data_rows)
    assert any(row.get("file_id") == "fileB" and row.get("center_ci_lo_mcse") in ("", None) for row in data_rows)
