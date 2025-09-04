import io
from core import data_io


def test_build_peak_table_includes_baseline_metadata_headers():
    # minimal record (values don't matter for header presence)
    rec = {
        "file": "a.csv",
        "peak": 1,
        "center": 0.0,
        "height": 1.0,
        "fwhm": 1.0,
        "eta": 0.5,
        "lock_width": False,
        "lock_center": False,
        "area": 1.0,
        "area_pct": 100.0,
        "rmse": 0.0,
        "fit_ok": True,
        "mode": "add",
        "als_lam": 1e5,
        "als_p": 1e-3,
        "baseline_method": "als",
        "poly_degree": float("nan"),
        "poly_normalize_x": float("nan"),
        "fit_xmin": 0.0,
        "fit_xmax": 1.0,
        "solver_choice": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_fscale": 1.0,
    }
    csv_txt = data_io.build_peak_table([rec])
    header = csv_txt.splitlines()[0]
    # Ensure new headers exist
    for col in ("baseline_method", "poly_degree", "poly_normalize_x"):
        assert col in header, f"missing {col} in peak-table header"
