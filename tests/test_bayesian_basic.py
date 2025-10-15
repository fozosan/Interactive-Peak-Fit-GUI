import numpy as np
import pandas as pd
import pytest
from core import fit_api, uncertainty, data_io


def test_bayesian_basic(two_peak_data, tmp_path):
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
        seed=123,
        n_steps=200,
        n_burn=100,
        thin=5,
        return_band=True,
    )
    assert res.get("method") != "NotAvailable"
    stats = res["param_stats"]
    assert np.any(stats["std"] > 0)
    # Bands are disabled for Bayesian uncertainty
    assert res.get("band") is None
    paths = data_io.derive_export_paths(str(tmp_path / "out.csv"))
    data_io.write_uncertainty_csv(paths["unc_csv"], res)
    data_io.write_uncertainty_txt(paths["unc_txt"], res)
    text = paths["unc_txt"].read_text(encoding="utf-8")
    assert "±" in text
    df2 = pd.read_csv(paths["unc_csv"])
    for pname in res.param_stats.keys():
        if pname == "rows":
            continue
        assert f"{pname}_sd" in df2.columns
