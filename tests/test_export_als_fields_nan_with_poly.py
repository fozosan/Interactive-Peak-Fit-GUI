import numpy as np
import pandas as pd
from core import fit_api, models, peaks, signals, data_io


def test_export_als_fields_nan_with_poly(tmp_path):
    x = np.linspace(-5, 5, 201)
    seed_peak = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    y = models.pv_sum(x, [seed_peak])
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "baseline": {"method": "polynomial", "degree": 2, "normalize_x": True},
        "baseline_uses_fit_range": True,
    }
    baseline = signals.polynomial_baseline(x, y, degree=2, normalize_x=True)
    res = fit_api.run_fit_consistent(x, y, [seed_peak], cfg, baseline, "add", mask)
    peaks_out = res["peaks_out"]
    areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in peaks_out]
    total = sum(areas) or 1.0
    records = []
    for i, (p, a) in enumerate(zip(peaks_out, areas), start=1):
        records.append(
            {
                "file": "sample",
                "peak": i,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "lock_width": p.lock_width,
                "lock_center": p.lock_center,
                "area": a,
                "area_pct": 100.0 * a / total,
                "rmse": res["rmse"],
                "fit_ok": res["fit_ok"],
                "mode": "add",
                "fit_xmin": x[0],
                "fit_xmax": x[-1],
                "solver_choice": "modern_vp",
                "use_baseline": True,
                "baseline_mode": "add",
                "baseline_uses_fit_range": True,
                "baseline_method": "polynomial",
                "als_lam": np.nan,
                "als_p": np.nan,
                "als_niter": np.nan,
                "als_thresh": np.nan,
                "poly_degree": 2,
                "poly_normalize_x": True,
            }
        )
    csv = data_io.build_peak_table(records)
    out = tmp_path / "peaks.csv"
    out.write_text(csv, encoding="utf-8")
    df = pd.read_csv(out)
    for col in ["als_lam", "als_p", "als_niter", "als_thresh"]:
        assert col in df.columns
        assert df[col].isna().all()
    # baseline metadata present and populated for polynomial
    assert (df["baseline_method"] == "polynomial").all()
    assert (df["poly_degree"] == 2).all()
    assert df["poly_normalize_x"].astype(bool).all()
