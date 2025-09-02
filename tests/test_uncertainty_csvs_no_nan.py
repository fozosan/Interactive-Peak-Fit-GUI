import csv, math, pathlib
from core import data_io


def _all_finite(row, keys):
    for k in keys:
        v = row.get(k, "")
        try:
            f = float(v)
        except Exception:
            return False
        if not math.isfinite(f):
            return False
    return True


def test_uncertainty_csvs_no_nan(tmp_path):
    # minimal normalized result with non-finite inputs that must be sanitized
    unc_norm = {
        "label": "Asymptotic",
        "method": "Asymptotic",
        "rmse": float("nan"),
        "dof": 0,
        "stats": [
            {"peak": 1, "param": "center", "value": float("nan"), "stderr": float("nan")},
            {"peak": 1, "param": "height", "value": 2.0, "stderr": float("nan")},
            {"peak": 1, "param": "fwhm", "value": 0.5, "stderr": 0.0, "p2_5": float("nan"), "p97_5": float("nan")},
        ],
    }
    base = tmp_path / "s"
    long_csv, wide_csv = data_io.write_uncertainty_csvs(base, "file.csv", unc_norm, write_wide=True)

    # long CSV: check all required numeric fields are finite
    with open(long_csv, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
    assert rows, "long CSV empty"
    for r in rows:
        assert _all_finite(r, ["value", "stderr", "p2_5", "p97_5", "rmse", "dof"])

    # wide CSV: check *_ci_lo/_ci_hi and values are finite where present
    if wide_csv:
        with open(wide_csv, newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)
        assert rows, "wide CSV empty"
        num_cols = [
            c
            for c in rdr.fieldnames
            if c
            and (c.endswith("_ci_lo") or c.endswith("_ci_hi") or c in ("center", "height", "fwhm", "eta"))
        ]
        for r in rows:
            for c in num_cols:
                if r.get(c, "") == "":
                    # allow eta if not present in stats
                    continue
                try:
                    f = float(r[c])
                except Exception:
                    assert False, f"non-numeric in wide CSV: {c}={r[c]!r}"
                assert math.isfinite(f), f"non-finite in wide CSV: {c}={r[c]!r}"

