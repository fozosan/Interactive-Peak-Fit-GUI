import pandas as pd, numpy as np, math, pathlib
from core import data_io


def test_uncertainty_csvs_no_nan(tmp_path):
    # Craft a minimal normalized result with NaNs that should be sanitized
    unc_norm = {
        "label": "Asymptotic",
        "method": "Asymptotic",
        "rmse": float("nan"),
        "dof": 0,
        "stats": [
            {"peak":1,"param":"center","value": float("nan"), "stderr": float("nan")},
            {"peak":1,"param":"height","value": 2.0, "stderr": float("nan")},
            {"peak":1,"param":"fwhm","value": 0.5, "stderr": 0.0, "p2_5": float("nan"), "p97_5": float("nan")},
        ],
    }
    base = tmp_path / "s"
    long_csv, wide_csv = data_io.write_uncertainty_csvs(base, "file.csv", unc_norm, write_wide=True)

    df_long = pd.read_csv(long_csv)
    assert not df_long[["value","stderr","p2_5","p97_5","rmse","dof"]].isna().any().any()

    if wide_csv:
        df_wide = pd.read_csv(wide_csv)
        num_cols = [c for c in df_wide.columns if c.endswith("_ci_lo") or c.endswith("_ci_hi") or c in ("center","height","fwhm")]
        assert not df_wide[num_cols].isna().any().any()
