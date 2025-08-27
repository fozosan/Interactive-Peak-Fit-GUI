"""Ensure single and batch exports match for the same spectrum."""

import numpy as np
import pandas as pd

from core import fit_api, models, peaks, signals
from batch import runner


def test_single_batch_export_parity(tmp_path):
    x = np.linspace(-5, 5, 101)
    pk = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    y = models.pv_sum(x, [pk])
    mask = np.ones_like(x, bool)

    fpath = tmp_path / "spec.csv"
    np.savetxt(fpath, np.column_stack([x, y]), delimiter=",")

    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "perf_seed_all": True,
    }
    baseline = signals.als_baseline(y, lam=1e5, p=0.001, niter=10, tol=0.0)
    res = fit_api.run_fit_consistent(x, y, [pk], cfg, baseline, "add", mask, rng_seed=123)
    records = []
    for i, p in enumerate(res["peaks_out"], start=1):
        records.append(
            {
                "file": "spec.csv",
                "peak": i,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "rmse": res["rmse"],
            }
        )
    single_df = pd.DataFrame(records)

    cfg_batch = {
        "peaks": [pk.__dict__],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "source": "template",
        "output_dir": str(tmp_path),
        "output_base": "batch",
    }
    runner.run([str(fpath)], cfg_batch)
    batch_df = pd.read_csv(tmp_path / "batch_fit.csv")

    cols = ["peak", "center", "height", "fwhm", "eta", "rmse"]
    pd.testing.assert_frame_equal(
        single_df[cols].reset_index(drop=True),
        batch_df[cols].reset_index(drop=True),
        rtol=1e-8,
        atol=1e-10,
    )

