import hashlib
import pandas as pd
from core import fit_api, uncertainty, data_io


def _export(path_base, theta, std):
    paths = data_io.derive_export_paths(str(path_base))
    df_fit = pd.DataFrame({"theta": theta})
    data_io.write_dataframe(df_fit, paths["fit"])
    df_unc = pd.DataFrame({"theta": theta, "se": std})
    data_io.write_dataframe(df_unc, paths["unc_csv"])
    return (
        hashlib.md5(paths["fit"].read_bytes()).hexdigest(),
        hashlib.md5(paths["unc_csv"].read_bytes()).hexdigest(),
        paths["fit"],
        paths["unc_csv"],
    )


def test_seeded_determinism(two_peak_data, tmp_path, no_blank_lines):
    res = fit_api.run_fit_consistent(**two_peak_data, return_jacobian=True)
    unc = uncertainty.asymptotic_ci(
        res["theta"], res["residual_fn"], res["jacobian"], res["ymodel_fn"]
    )
    h1_fit, h1_unc, p1_fit, p1_unc = _export(tmp_path / "a.csv", res["theta"], unc["param_std"])

    res2 = fit_api.run_fit_consistent(**two_peak_data, return_jacobian=True)
    unc2 = uncertainty.asymptotic_ci(
        res2["theta"], res2["residual_fn"], res2["jacobian"], res2["ymodel_fn"]
    )
    h2_fit, h2_unc, p2_fit, p2_unc = _export(tmp_path / "b.csv", res2["theta"], unc2["param_std"])

    assert h1_fit == h2_fit
    assert h1_unc == h2_unc
    assert no_blank_lines(p1_fit)
    assert no_blank_lines(p1_unc)
    assert no_blank_lines(p2_fit)
    assert no_blank_lines(p2_unc)
