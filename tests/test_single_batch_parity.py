"""Verify single and batch fits yield matching results."""

import numpy as np
import pandas as pd

from core import fit_api, models, peaks, signals
from batch import runner


def _build_baseline(x, y, mask, use_slice):
    if use_slice and np.any(mask) and not np.all(mask):
        x_sub = x[mask]
        y_sub = y[mask]
        z_sub = signals.als_baseline(y_sub, lam=1e5, p=0.001, niter=10, tol=0.0)
        return np.interp(x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
    return signals.als_baseline(y, lam=1e5, p=0.001, niter=10, tol=0.0)


def test_single_batch_parity(tmp_path):
    x = np.linspace(-5, 5, 201)
    seeds = [
        peaks.Peak(-0.8, 1.0, 0.6, 0.5),
        peaks.Peak(0.9, 0.8, 0.5, 0.4),
    ]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)

    fpath = tmp_path / "spec.csv"
    np.savetxt(fpath, np.column_stack([x, y]), delimiter=",")
    seed = abs(hash(str(fpath.resolve()))) & 0xFFFFFFFF

    for use_range in (False, True):
        baseline = _build_baseline(x, y, mask, use_range)
        cfg_single = {
            "solver": "modern_vp",
            "solver_loss": "linear",
            "solver_weight": "none",
            "perf_seed_all": True,
            "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
            "baseline_uses_fit_range": use_range,
        }
        res_single = fit_api.run_fit_consistent(
            x,
            y,
            seeds,
            cfg_single,
            baseline,
            "add",
            mask,
            rng_seed=seed,
        )

        cfg_batch = {
            "peaks": [p.__dict__ for p in seeds],
            "solver": "modern_vp",
            "mode": "add",
            "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
            "baseline_uses_fit_range": use_range,
            "output_dir": str(tmp_path),
            "output_base": "batch",
            "save_traces": False,
            "perf_seed_all": True,
        }
        runner.run([str(fpath)], cfg_batch)

        df = pd.read_csv(tmp_path / "batch_fit.csv").sort_values("peak")
        theta_batch = []
        for row in df.itertuples():
            theta_batch.extend([row.center, row.height, row.fwhm, row.eta])
        theta_batch = np.asarray(theta_batch)
        rmse_batch = df["rmse"].iloc[0]

        theta_single = res_single["theta"]
        rmse_single = res_single["rmse"]

        y_norm = np.linalg.norm(y)
        assert abs(rmse_single - rmse_batch) <= 1e-6 * max(1.0, y_norm)

        for i, (s, b) in enumerate(zip(theta_single, theta_batch)):
            delta = abs(s - b)
            if i % 4 == 0:  # centers allow absolute tolerance
                assert delta <= 1e-3 or delta / max(1e-12, abs(s)) <= 1e-4
            else:
                assert delta / max(1e-12, abs(s)) <= 1e-4

