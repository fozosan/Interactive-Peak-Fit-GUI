import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

def _write_synthetic(tmpdir: Path, name: str = "synth.csv") -> Path:
    x = np.linspace(0, 100, 201)
    # simple smooth baseline + single peak
    baseline = 0.01 * (x - 50.0)
    peak = 5.0 * np.exp(-0.5 * ((x - 40.0)/5.0)**2)
    y = baseline + peak + 0.05 * np.random.RandomState(0).randn(x.size)
    p = tmpdir / name
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for xi, yi in zip(x, y):
            w.writerow([xi, yi])
    return p

def _read_fit_rows(outdir: Path, stem: str):
    fit_path = outdir / f"{stem}_fit.csv"
    assert fit_path.exists(), f"missing {fit_path}"
    with fit_path.open("r", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    return rows

def _is_nan_str(v):
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("nan", "")

@pytest.mark.parametrize("method", ["als", "polynomial"])
def test_cli_baseline_method_exports_metadata(tmp_path, method):
    inp = _write_synthetic(tmp_path, "one.csv")
    outdir = tmp_path / "out"
    outdir.mkdir()

    cmd = [
        sys.executable, "-m", "tools.batch_cli",
        "--patterns", str(inp),
        "--outdir", str(outdir),
        "--solver", "classic",
        "--baseline-method", method,
        "--auto-max", "2",
    ]
    if method == "als":
        cmd += ["--als-lam", "200000", "--als-p", "0.002", "--als-niter", "12", "--als-thresh", "0.0"]
    else:
        cmd += ["--poly-degree", "3", "--poly-normalize-x"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    rows = _read_fit_rows(outdir, "one")
    assert rows, "no rows exported"
    row0 = rows[0]

    assert row0["baseline_method"] == method

    if method == "als":
        # ALS fields populated, poly fields NaN
        assert float(row0["als_lam"]) == 200000.0
        assert float(row0["als_p"]) == 0.002
        assert int(float(row0["als_niter"])) == 12
        # threshold may format as float
        assert float(row0["als_thresh"]) == 0.0
        assert _is_nan_str(row0.get("poly_degree"))
        assert _is_nan_str(row0.get("poly_normalize_x"))
    else:
        # Poly fields populated, ALS fields NaN
        assert int(float(row0["poly_degree"])) == 3
        # normalize_x stored as boolean -> may serialize as True/False
        assert str(row0["poly_normalize_x"]).lower() in ("true", "1")
        assert _is_nan_str(row0.get("als_lam"))
        assert _is_nan_str(row0.get("als_p"))
        assert _is_nan_str(row0.get("als_niter"))
        assert _is_nan_str(row0.get("als_thresh"))
