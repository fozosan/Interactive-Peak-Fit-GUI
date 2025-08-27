#!/usr/bin/env python
"""Helper to execute a single batch item in isolation.

This script is designed for smoke tests where native libraries with different
threading models may otherwise conflict when processing multiple spectra in a
single process. It enforces headless operation and CPU-only math libraries.
"""

import os
import argparse

# Headless + single-thread hardening before heavy imports
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("SMOKE_MODE", "1")

from batch.runner import run_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--filename", required=True)
    args = p.parse_args()

    run_batch(
        input_dir=args.indir,
        pattern=args.pattern,
        output_dir=args.outdir,
        source_mode="auto",
        reheight=False,
        seed=args.seed,
        workers=0,
        perf_overrides={"perf_gpu": False, "perf_numba": False},
        files_filter=[args.filename],
    )


if __name__ == "__main__":  # pragma: no cover - manual use
    main()

