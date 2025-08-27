#!/usr/bin/env python
"""Tiny smoke test for the uncertainty helpers."""
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.data_io import load_xy
from core.fit_api import run_fit_consistent
from core.peaks import Peak
from core.uncertainty import asymptotic_ci, bootstrap_ci


p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--method", choices=["asymptotic", "bootstrap"], default="asymptotic")
p.add_argument("--seed", type=int, default=123)
args = p.parse_args()

x, y = load_xy(args.file)
mask = np.ones_like(x, bool)
# crude single-peak seed based on maximum
center = float(x[np.argmax(y)])
height = float(y.max() - y.min())
width = float((x.max() - x.min()) / 10.0)
seeds = [Peak(center, height, width, 0.5)]
cfg = {"solver": "modern_vp", "solver_loss": "linear", "solver_weight": "none"}

data = {
    "x": x,
    "y": y,
    "peaks_in": seeds,
    "cfg": cfg,
    "baseline": None,
    "mode": "add",
    "fit_mask": mask,
    "rng_seed": args.seed,
}

fit = run_fit_consistent(**data, return_jacobian=True)
if args.method == "asymptotic":
    res = asymptotic_ci(
        fit["theta"], fit["residual_fn"], fit["jacobian"], fit["ymodel_fn"],
        alpha=0.05, svd_rcond=1e-10, grad_mode="complex"
    )
else:
    res = bootstrap_ci(
        engine=run_fit_consistent,
        data=data,
        n=40,
        band_percentiles=(2.5, 97.5),
        workers=0,
        seed_root=args.seed,
    )
print("OK:", "params=", res.get("param_mean"), "std=", res.get("param_std"))
