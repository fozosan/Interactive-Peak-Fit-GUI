import numpy as np
import math
from pathlib import Path
from core import fit_api, uncertainty, data_io
from tests.conftest import _maybe_read_unc_files, _pivot_long_to_wide


def test_unc_bootstrap_outputs(two_peak_data, tmp_path, monkeypatch):
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )

    def fake_run_fit_consistent(x, y, cfg, theta_init=None, locked_mask=None, bounds=None, baseline=None):
        locked_mask = np.asarray(locked_mask, bool)
        th = np.asarray(theta_init, float).copy()
        mean_y = float(np.mean(y))
        th[~locked_mask] += mean_y * 0.01
        return {"fit_ok": True, "theta": th}

    monkeypatch.setattr(fit_api, "run_fit_consistent", fake_run_fit_consistent)

    args = dict(
        theta=fit["theta"],
        residual=fit["residual"],
        jacobian=fit["jacobian"],
        predict_full=fit["predict_full"],
        x_all=fit["x"],
        y_all=two_peak_data["y"],
        bounds=fit.get("bounds"),
        param_names=fit.get("param_names"),
        locked_mask=fit.get("locked_mask"),
        fit_ctx=fit,
        n_boot=20,
        seed=42,
        workers=0,
    )
    res1 = uncertainty.bootstrap_ci(**args)
    assert res1.method_label == "Bootstrap"

    base = Path(tmp_path / "out.csv")
    unc_norm = data_io.normalize_unc_result(res1)
    for row in unc_norm["stats"]:
        for pname in ("center", "height", "fwhm", "eta"):
            p = row[pname]
            assert math.isfinite(p.get("est", float("nan")))
            assert math.isfinite(p.get("sd", float("nan")))
    data_io.write_uncertainty_csvs(base, "", unc_norm, write_wide=True)
    basedir = Path(tmp_path)
    stem = "out"
    wide_df, long_df, used = _maybe_read_unc_files(basedir, stem)
    if wide_df is None:
        assert long_df is not None, "No uncertainty CSVs were written"
        wide_df = _pivot_long_to_wide(long_df)

    assert not wide_df.empty
    assert set(["file","peak","method"]).issubset(wide_df.columns)
    methods = wide_df["method"].astype(str).str.lower()
    assert any(methods.str.startswith("bootstrap"))

    for col in ["center","center_stderr","height","height_stderr"]:
        assert col in wide_df.columns, f"missing column {col}"
        assert str(wide_df[col].dtype).startswith(("float","int","UInt")), f"{col} must be numeric"

    res2 = uncertainty.bootstrap_ci(**args)
    assert np.allclose(res1["param_mean"], res2["param_mean"])
    assert np.allclose(res1["param_std"], res2["param_std"])
