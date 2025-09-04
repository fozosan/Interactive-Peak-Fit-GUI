from pathlib import Path
import numpy as np
import csv
from batch import runner


def _write_xy(path: Path, x, y):
    with path.open("w", encoding="utf-8") as fh:
        for xi, yi in zip(x, y):
            fh.write(f"{xi},{yi}\n")


def _read_first_row(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        row = next(rdr)
    idx = {name: i for i, name in enumerate(header)}
    return header, row, idx


def _simple_signal():
    x = np.linspace(0, 100, 201)
    # simple Gaussian-ish peak
    x0 = 50.0
    fwhm = 10.0
    A = 100.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y = A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return x, y


def _one_peak(center, height=50.0, fwhm=12.0, eta=0.5):
    # Shape expected by batch config ("peaks" list of dicts)
    return {
        "center": float(center),
        "height": float(height),
        "fwhm": float(fwhm),
        "eta": float(eta),
        "lock_center": False,
        "lock_width": False,
    }


def test_batch_csv_has_baseline_method_and_fields_for_als(tmp_path):
    # Prepare data
    x, y = _simple_signal()
    f = tmp_path / "s1.csv"
    _write_xy(f, x, y)

    cfg = {
        "peaks": [_one_peak(50.0)],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"method": "als", "lam": 1e5, "p": 1e-3, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 1,
        "baseline_uses_fit_range": True,
        "output_dir": str(tmp_path),
        "output_base": "out",
    }

    runner.run_batch([str(f)], cfg, compute_uncertainty=False)
    fit_csv = tmp_path / f"{f.stem}_fit.csv"
    assert fit_csv.exists(), "fit CSV not written"

    header, row, idx = _read_first_row(fit_csv)
    assert "baseline_method" in header and "poly_degree" in header and "poly_normalize_x" in header
    assert row[idx["baseline_method"]].strip().lower() == "als"
    # ALS numeric fields should be finite
    assert row[idx["als_lam"]] not in ("", "nan", "NaN")
    assert row[idx["als_p"]] not in ("", "nan", "NaN")
    # Polynomial-only fields should be NaN when ALS selected
    assert row[idx["poly_degree"]].lower() == "nan"
    assert row[idx["poly_normalize_x"]].lower() == "nan"


def test_batch_csv_has_baseline_method_and_fields_for_polynomial(tmp_path):
    # Prepare data
    x, y = _simple_signal()
    f = tmp_path / "s2.csv"
    _write_xy(f, x, y)

    cfg = {
        "peaks": [_one_peak(50.0)],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"method": "polynomial", "degree": 2, "normalize_x": True},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 1,
        "baseline_uses_fit_range": True,
        "output_dir": str(tmp_path),
        "output_base": "out2",
    }

    runner.run_batch([str(f)], cfg, compute_uncertainty=False)
    fit_csv = tmp_path / f"{f.stem}_fit.csv"
    assert fit_csv.exists(), "fit CSV not written"

    header, row, idx = _read_first_row(fit_csv)
    assert "baseline_method" in header and "poly_degree" in header and "poly_normalize_x" in header
    assert row[idx["baseline_method"]].strip().lower() == "polynomial"
    # ALS-only fields should be NaN under polynomial baseline
    assert row[idx["als_lam"]].lower() == "nan"
    assert row[idx["als_p"]].lower() == "nan"
    # Polynomial fields populated
    assert row[idx["poly_degree"]] not in ("", "nan", "NaN")
    # bools serialize as 'True'/'False'
    assert row[idx["poly_normalize_x"]].strip() in ("True", "False")
