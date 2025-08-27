import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import models, peaks, signals, fit_api
from batch import runner


@pytest.mark.parametrize(
    "mode,loss,weight,restarts",
    list(
        itertools.product(
            ["add", "subtract"],
            ["linear", "soft_l1"],
            ["none", "poisson"],
            [1, 3],
        )
    ),
)
def test_batch_single_parity(mode, loss, weight, restarts, tmp_path):
    x = np.linspace(-5, 5, 201)
    seeds = [
        peaks.Peak(-1.0, 1.0, 0.8, 0.2, lock_center=True),
        peaks.Peak(1.0, 0.6, 0.5, 0.3, lock_width=True),
    ]
    baseline_true = 0.1 * x + 0.2
    y_raw = baseline_true + models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    base = signals.als_baseline(y_raw, lam=1e5, p=0.001, niter=10, tol=0.0)

    cfg = {
        "solver": "modern_vp",
        "solver_loss": loss,
        "solver_weight": weight,
        "solver_fscale": 1.0,
        "solver_maxfev": 100,
        "solver_restarts": restarts,
        "solver_jitter_pct": 10,
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
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
        base,
        mode,
        mask,
        rng_seed=seed,
    )

    cfg_batch = {
        **cfg,
        "peaks": [p.__dict__ for p in seeds],
        "mode": mode,
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

    theta_single = res_single["theta"]
    rmse_single = res_single["rmse"]

    assert abs(rmse_batch - rmse_single) <= 1e-8
    delta = np.abs(theta_batch - theta_single)
    rel = delta / np.maximum(1e-12, np.abs(theta_single))
    assert np.all((delta <= 1e-8) | (rel <= 1e-6))
    for i, pk in enumerate(seeds):
        if pk.lock_center:
            assert abs(theta_batch[4 * i] - pk.center) <= 1e-12
        if pk.lock_width:
            assert abs(theta_batch[4 * i + 2] - pk.fwhm) <= 1e-12

