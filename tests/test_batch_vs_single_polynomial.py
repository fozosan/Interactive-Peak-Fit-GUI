import numpy as np
import pandas as pd
from core import fit_api, models, peaks, signals
from batch import runner


def test_batch_vs_single_polynomial(tmp_path):
    x = np.linspace(-5, 5, 201)
    seeds = [
        peaks.Peak(-0.8, 1.0, 0.6, 0.5),
        peaks.Peak(0.9, 0.8, 0.5, 0.4),
    ]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    fpath = tmp_path / "spec.csv"
    np.savetxt(fpath, np.column_stack([x, y]), delimiter=",")

    cfg_single = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "perf_seed_all": True,
        "baseline": {"method": "polynomial", "degree": 3, "normalize_x": True},
        "baseline_uses_fit_range": True,
    }
    baseline = signals.polynomial_baseline(x, y, degree=3, normalize_x=True)
    res_single = fit_api.run_fit_consistent(
        x,
        y,
        seeds,
        cfg_single,
        baseline,
        "add",
        mask,
        rng_seed=123,
    )

    cfg_batch = {
        "peaks": [p.__dict__ for p in seeds],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"method": "polynomial", "degree": 3, "normalize_x": True},
        "baseline_uses_fit_range": True,
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

    assert np.allclose(res_single["theta"], theta_batch, atol=1e-6, rtol=1e-6)
    assert np.isnan(df["als_lam"]).all()
    assert (tmp_path / f"{fpath.stem}_uncertainty.txt").exists()
