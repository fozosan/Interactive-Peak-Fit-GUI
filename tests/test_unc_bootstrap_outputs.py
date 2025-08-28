import numpy as np
import pandas as pd
from core import fit_api, uncertainty, data_io


def test_unc_bootstrap_outputs(two_peak_data, tmp_path):
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    res1 = uncertainty.bootstrap_ci(fit, n_boot=20, seed=42, workers=0)
    assert res1.method_label == "Bootstrap (residual)"

    paths = data_io.derive_export_paths(str(tmp_path / "out.csv"))
    data_io.write_uncertainty_csv(paths["unc_csv"], res1)
    df = pd.read_csv(paths["unc_csv"])
    for pname in res1.params.keys():
        sd_col = f"{pname}_sd"
        assert sd_col in df.columns
        assert np.isfinite(df.loc[0, sd_col])
        p_lo = f"{pname}_p2_5"
        p_hi = f"{pname}_p97_5"
        assert p_lo in df.columns and p_hi in df.columns
        est = df.loc[0, f"{pname}_est"]
        if np.isfinite(df.loc[0, p_lo]) and np.isfinite(df.loc[0, p_hi]):
            assert df.loc[0, p_lo] - 1e-9 <= est <= df.loc[0, p_hi] + 1e-9

    res2 = uncertainty.bootstrap_ci(fit, n_boot=20, seed=42, workers=0)
    assert np.allclose(res1["param_mean"], res2["param_mean"])
    assert np.allclose(res1["param_std"], res2["param_std"])
