"""Ensure bounds and locks are honoured in batch fits."""

import numpy as np
import pandas as pd

from batch import runner
from core import models, peaks


def test_bounds_and_locks_enforced(tmp_path):
    x = np.linspace(-4, 4, 101)
    seed = peaks.Peak(0.0, 1.0, 1.0, 0.5, lock_center=True, lock_width=True)
    y = models.pv_sum(x, [seed])

    fpath = tmp_path / "s.csv"
    np.savetxt(fpath, np.column_stack([x, y]), delimiter=",")

    cfg = {
        "peaks": [seed.__dict__],
        "solver": "modern_vp",
        "mode": "add",
        "solver_loss": "linear",
        "solver_weight": "none",
        "modern_vp": {"min_fwhm": 0.9, "max_fwhm": 1.1, "bound_centers_to_window": True},
        "output_dir": str(tmp_path),
        "output_base": "out",
    }

    runner.run([str(fpath)], cfg)

    df = pd.read_csv(tmp_path / "out_fit.csv")
    row = df.iloc[0]

    assert 0.9 <= row["fwhm"] <= 1.1
    assert abs(row["center"] - seed.center) <= 1e-12
    assert abs(row["fwhm"] - seed.fwhm) <= 1e-12
    assert row["height"] >= 0

