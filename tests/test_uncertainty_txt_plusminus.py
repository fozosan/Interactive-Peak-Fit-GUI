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

    data_io.write_uncertainty_csv(paths["unc_csv"], unc)
    data_io.write_uncertainty_txt(paths["unc_txt"], unc)

    text = paths["unc_txt"].read_text(encoding="utf-8")
    for pname in ("p0", "p1", "p2", "p3"):
        assert f"{pname} =" in text and "±" in text

    df2 = pd.read_csv(paths["unc_csv"])
    assert set(df2.columns) == {"param", "mean", "std", "q05", "q50", "q95", "method", "ess", "rhat"}

