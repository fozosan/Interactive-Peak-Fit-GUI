import pandas as pd
from core import fit_api, data_io, uncertainty


def test_export_filenames_and_noblanks(two_peak_data, tmp_path, no_blank_lines):
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    paths = data_io.derive_export_paths(str(tmp_path / "spec.csv"))
    # fit csv
    df_fit = pd.DataFrame({"theta": fit["theta"]})
    data_io.write_dataframe(df_fit, paths["fit"])
    # trace csv (minimal)
    trace_csv = data_io.build_trace_table(two_peak_data["x"], two_peak_data["y"], None, [])
    paths["trace"].write_text(trace_csv, encoding="utf-8")
    # uncertainty
    unc = uncertainty.asymptotic_ci(
        fit["theta"], fit["residual_fn"], fit["jacobian"], fit["ymodel_fn"]
    )
    data_io.write_uncertainty_csv(paths["unc_csv"], unc)
    data_io.write_uncertainty_txt(paths["unc_txt"], unc)

    for key in ("fit", "trace", "unc_csv", "unc_txt"):
        assert paths[key].exists()
    for key in ("fit", "trace", "unc_csv"):
        assert no_blank_lines(paths[key])

    df = pd.read_csv(paths["unc_csv"])
    for pname in unc.param_stats.keys():
        assert f"{pname}_sd" in df.columns
    text = paths["unc_txt"].read_text(encoding="utf-8")
    assert "±" in text
