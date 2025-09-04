from __future__ import annotations
import argparse
import sys
from pathlib import Path

from batch.runner import run_batch

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal batch CLI wrapper for Peakfit.")
    p.add_argument("--patterns", required=True, help="Glob(s) for input files; separate multiple with ';'")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--solver", default="classic", help="Solver backend (classic|modern_vp|modern_trf|lmfit_vp)")
    p.add_argument("--mode", default="add", choices=["add", "subtract"], help="Baseline mode")
    # Baseline switch
    p.add_argument("--baseline-method", default="als", choices=["als", "polynomial"], help="Baseline method")
    # ALS options (kept for compatibility)
    p.add_argument("--als-lam", type=float, default=1e5)
    p.add_argument("--als-p", type=float, default=0.001)
    p.add_argument("--als-niter", type=int, default=10)
    p.add_argument("--als-thresh", type=float, default=0.0)
    # Polynomial options
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--poly-normalize-x", dest="poly_normalize_x", action="store_true", default=True)
    p.add_argument("--no-poly-normalize-x", dest="poly_normalize_x", action="store_false")
    # Fit-range baseline toggling
    p.add_argument("--baseline-uses-fit-range", dest="baseline_uses_fit_range", action="store_true", default=True)
    p.add_argument("--no-baseline-uses-fit-range", dest="baseline_uses_fit_range", action="store_false")
    # Seeding: keep robust for tests
    p.add_argument("--source", default="auto", choices=["auto", "current", "template"])
    p.add_argument("--auto-max", type=int, default=3)
    p.add_argument("--reheight", action="store_true", default=False)
    # Uncertainty: default off for speed
    p.add_argument("--uncertainty", action="store_true", default=False)
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    patterns = [s.strip() for s in args.patterns.split(";") if s.strip()]
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = {"method": args.baseline_method}
    if args.baseline_method == "als":
        baseline.update({"lam": args.als_lam, "p": args.als_p, "niter": args.als_niter, "thresh": args.als_thresh})
    else:
        baseline.update({"degree": int(args.poly_degree), "normalize_x": bool(args.poly_normalize_x)})

    cfg = {
        "peaks": [],                     # seeding controlled by source/auto
        "solver": args.solver,
        "mode": args.mode,
        "baseline": baseline,
        "baseline_uses_fit_range": bool(args.baseline_uses_fit_range),
        "save_traces": False,
        "source": args.source,
        "reheight": bool(args.reheight),
        "auto_max": int(args.auto_max),
        "output_dir": str(out_dir),
        "output_base": "batch",
        # keep predictable/simple for tests
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": False,
        "perf_seed_all": True,
        "perf_max_workers": 0,
    }

    compute_unc = bool(args.uncertainty)
    ok, processed = run_batch(patterns, cfg, compute_uncertainty=compute_unc)
    return 0 if ok > 0 and processed > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
