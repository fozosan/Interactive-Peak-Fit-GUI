from core import fit_api, uncertainty, data_io
import pytest


def test_bayesian_optional(two_peak_data, tmp_path):
    pytest.importorskip("emcee")
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    res = uncertainty.bayesian_ci(
        fit["theta"],
        predict_full=fit["predict_full"],
        x_all=two_peak_data["x"],
        y_all=two_peak_data["y"],
        residual_fn=fit["residual_fn"],
        locked_mask=fit.get("locked_mask"),
        n_steps=20,
        n_burn=10,
        seed=1,
        return_band=False,
    )
    paths = data_io.derive_export_paths(str(tmp_path / "out.csv"))
    data_io.write_uncertainty_csv(paths["unc_csv"], res)
    data_io.write_uncertainty_txt(paths["unc_txt"], res)
    text = paths["unc_txt"].read_text(encoding="utf-8")
    assert "±" in text
