import hashlib
import pandas as pd
import hashlib
import numpy as np
from core import fit_api, uncertainty, data_io
from core.uncertainty import UncertaintyResult


def _export(path_base, theta, std):
    paths = data_io.derive_export_paths(str(path_base))
    df_fit = pd.DataFrame({"theta": theta})
    data_io.write_dataframe(df_fit, paths["fit"])
    param_stats = {
        f"p{i}": {
            "est": float(theta[i]),
            "sd": float(std[i]),
            "p2.5": float(theta[i] - 1.96 * std[i]),
            "p97.5": float(theta[i] + 1.96 * std[i]),
        }
        for i in range(len(theta))
    }
    unc = UncertaintyResult("asymptotic", None, param_stats, {})
    data_io.write_uncertainty_csv(paths["unc_csv"], unc)
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

    # Bayesian determinism when backend available
    fit = fit_api.run_fit_consistent(
        **two_peak_data, return_jacobian=True, return_predictors=True
    )
    try:
        b1 = uncertainty.bayesian_ci(fit, seed=123, n_steps=20, n_burn=10, n_walkers=8)
        b2 = uncertainty.bayesian_ci(fit, seed=123, n_steps=20, n_burn=10, n_walkers=8)
    except ImportError:
        return
    assert np.allclose(b1["param_mean"], b2["param_mean"])
    assert np.allclose(b1["param_std"], b2["param_std"])
