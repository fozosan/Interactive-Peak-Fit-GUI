import csv, math
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


def test_batch_summary_no_nan(tmp_path):
    rows = [
        {
            "file": "a.csv",
            "peak": 1,
            "param": "center",
            "value": float("nan"),
            "stderr": float("nan"),
            "method": "Asymptotic",
            "rmse": float("nan"),
            "dof": 0,
        },
        {
            "file": "a.csv",
            "peak": 1,
            "param": "height",
            "value": 3.0,
            "stderr": 0.0,
            "p2_5": float("nan"),
            "p97_5": float("nan"),
            "method": "Asymptotic",
            "rmse": 0.1,
            "dof": 0,
        },
    ]
    long_path, legacy_path = data_io.write_batch_uncertainty_long(tmp_path, rows)

    for p in (long_path, legacy_path):
        with open(p, newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            out_rows = list(rdr)
        assert out_rows, f"{p.name} empty"
        for r in out_rows:
            assert _all_finite(r, ["value", "stderr", "p2_5", "p97_5", "rmse", "dof"])

