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
from core.uncertainty import asymptotic_ci, bootstrap_ci, bayesian_ci, NotAvailable


p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument(
    "--method", choices=["asymptotic", "bootstrap", "bayesian"], default="asymptotic"
)
p.add_argument("--seed", type=int, default=123)
p.add_argument("--nboot", type=int, default=40)
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

fit = run_fit_consistent(**data, return_jacobian=True, return_predictors=True)
if args.method == "asymptotic":
    res = asymptotic_ci(
        fit["theta"], fit["residual_fn"], fit["jacobian"], fit["ymodel_fn"]
    )
elif args.method == "bootstrap":
    res = bootstrap_ci(fit, n_boot=args.nboot, seed=args.seed, workers=0)
else:
    res = bayesian_ci(fit, seed=args.seed, n_steps=args.nboot)

if isinstance(res, NotAvailable):
    print("Bayesian not available:", res.reason)
    sys.exit(0)

df = res["param_stats"]
df.to_csv("uncertainty.csv", index=False)
print(df.head())
band = res.get("band")
if band is not None:
    x, lo, hi = band
    print("band min/max:", float(lo.min()), float(hi.max()))
