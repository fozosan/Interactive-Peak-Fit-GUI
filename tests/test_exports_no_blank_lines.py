import pandas as pd
from core import fit_api, data_io, uncertainty


def test_exports_no_blank_lines(two_peak_data, tmp_path, no_blank_lines):
    res = fit_api.run_fit_consistent(**two_peak_data, return_jacobian=True)
    paths = data_io.derive_export_paths(str(tmp_path / "spec.csv"))
    df_fit = pd.DataFrame({"theta": res["theta"]})
    data_io.write_dataframe(df_fit, paths["fit"])
    unc = uncertainty.asymptotic_ci(res["theta"], res["residual_fn"], res["jacobian"], res["ymodel_fn"])
    data_io.write_uncertainty_csv(paths["unc_csv"], unc)
    assert no_blank_lines(paths["fit"])
    assert no_blank_lines(paths["unc_csv"])
