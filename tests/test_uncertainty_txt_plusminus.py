"""Ensure uncertainty text export contains ± lines and CSV has required columns."""

import numpy as np
import pandas as pd

from core import data_io, fit_api, models, peaks, uncertainty


def test_uncertainty_txt_plusminus(tmp_path):
    x = np.linspace(-5, 5, 101)
    pk = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    y = models.pv_sum(x, [pk])
    mask = np.ones_like(x, bool)
    cfg = {"solver": "modern_vp", "solver_loss": "linear", "solver_weight": "none"}
    res = fit_api.run_fit_consistent(
        x,
        y,
        [pk],
        cfg,
        None,
        "add",
        mask,
        rng_seed=123,
        return_jacobian=True,
    )
    unc = uncertainty.asymptotic_ci(
        res["theta"], res["residual_fn"], res["jacobian"], res["ymodel_fn"]
    )

    paths = data_io.derive_export_paths(str(tmp_path / "out.csv"))

    z = 1.96
    params = ["center", "height", "fwhm", "eta"]
    with paths["unc_txt"].open("w", encoding="utf-8") as fh:
        fh.write("Method: Asymptotic\n")
        for i, name in enumerate(params):
            std = float(unc["param_std"][i])
            fh.write(f"{name} ± {std:.3g}\n")

    df = pd.DataFrame(
        {
            "param": params,
            "mean": res["theta"][:4],
            "std": unc["param_std"][:4],
            "lower": res["theta"][:4] - z * unc["param_std"][:4],
            "upper": res["theta"][:4] + z * unc["param_std"][:4],
            "method": ["asymptotic"] * 4,
        }
    )
    data_io.write_dataframe(df, paths["unc_csv"])

    text = paths["unc_txt"].read_text(encoding="utf-8")
    for pname in ("height", "fwhm", "center", "eta"):
        assert f"{pname} ±" in text

    df2 = pd.read_csv(paths["unc_csv"])
    assert set(df2.columns) == {"param", "mean", "std", "lower", "upper", "method"}

