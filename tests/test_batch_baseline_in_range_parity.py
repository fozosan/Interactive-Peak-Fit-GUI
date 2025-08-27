from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import models, peaks, signals, fit_api
from batch import runner


def test_batch_baseline_in_range_parity(tmp_path):
    x = np.linspace(-5, 5, 201)
    base_true = 0.1 * x + 0.2
    seeds = [peaks.Peak(-1.0, 1.0, 0.8, 0.2)]
    y_raw = base_true + models.pv_sum(x, seeds)

    mask = (x >= -2) & (x <= 2)
    z_sub = signals.als_baseline(y_raw[mask], lam=1e5, p=0.01, niter=10, tol=0.0)
    baseline_full = np.interp(x, x[mask], z_sub, left=z_sub[0], right=z_sub[-1])

    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_maxfev": 100,
        "solver_restarts": 1,
        "solver_jitter_pct": 0,
        "baseline": {"lam": 1e5, "p": 0.01, "niter": 10, "thresh": 0.0},
        "perf_seed_all": True,
    }

    fpath = tmp_path / "spec.csv"
    np.savetxt(fpath, np.column_stack([x, y_raw]), delimiter=",")
    seed = abs(hash(str(fpath.resolve()))) & 0xFFFFFFFF

    res_single = fit_api.run_fit_consistent(
        x,
        y_raw,
        seeds,
        cfg,
        baseline_full,
        "add",
        mask,
        rng_seed=seed,
    )

    cfg_batch = {
        **cfg,
        "peaks": [p.__dict__ for p in seeds],
        "mode": "add",
        "baseline_uses_fit_range": True,
        "fit_xmin": -2,
        "fit_xmax": 2,
        "output_dir": str(tmp_path),
        "output_base": "out",
        "save_traces": False,
    }
    runner.run([str(fpath)], cfg_batch, progress=None, log=None)

    df = pd.read_csv(tmp_path / "out_fit.csv").sort_values("peak")
    theta_batch = []
    for _, row in df.iterrows():
        theta_batch.extend([row["center"], row["height"], row["fwhm"], row["eta"]])
    theta_batch = np.array(theta_batch)
    rmse_batch = df["rmse"].iloc[0]

    assert np.allclose(theta_batch, res_single["theta"], rtol=1e-6, atol=1e-8)
    assert abs(rmse_batch - res_single["rmse"]) <= 1e-8

    z_sub_batch = signals.als_baseline(y_raw[mask], lam=1e5, p=0.01, niter=10, tol=0.0)
    baseline_full_batch = np.interp(x, x[mask], z_sub_batch, left=z_sub_batch[0], right=z_sub_batch[-1])
    assert np.linalg.norm(baseline_full_batch - baseline_full) < 1e-10

