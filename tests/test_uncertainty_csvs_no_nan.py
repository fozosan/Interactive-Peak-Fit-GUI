import csv, math, pathlib
from core import data_io


def _finite_or_blank(row, keys):
    """Every cell is a finite number or blank: never "nan", never a made-up 0.0."""
    for k in keys:
        v = row.get(k, "")
        if v == "":
            continue
        try:
            f = float(v)
        except Exception:
            return False
        if not math.isfinite(f):
            return False
    return True


def test_uncertainty_csvs_no_nan(tmp_path):
    # minimal normalized result with non-finite inputs: they are written blank
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
    long_csv, wide_csv = data_io.write_uncertainty_csvs(
        base, "file.csv", data_io.normalize_unc_result(unc_norm), write_wide=True
    )

    with open(long_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "long CSV empty"
    for r in rows:
        assert _finite_or_blank(r, ["value", "stderr", "p2_5", "p97_5", "rmse", "dof"])
        # a NaN rmse and dof 0 are missing, not 0.0 and 1
        assert r["rmse"] == "" and r["dof"] == ""
    by = {r["param"]: r for r in rows}
    assert by["center"]["value"] == by["center"]["stderr"] == by["center"]["p2_5"] == ""
    assert float(by["height"]["value"]) == 2.0 and by["height"]["stderr"] == ""
    # a zero standard error is a value: the interval collapses onto the estimate
    assert float(by["fwhm"]["stderr"]) == 0.0 and float(by["fwhm"]["p2_5"]) == 0.5

    if wide_csv:
        with open(wide_csv, newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)
        assert rows, "wide CSV empty"
        for r in rows:
            assert _finite_or_blank(r, rdr.fieldnames[3:])
