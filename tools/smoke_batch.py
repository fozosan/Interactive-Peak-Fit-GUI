#!/usr/bin/env python
"""Robust smoke test for the batch runner.

Each input file is processed in a fresh subprocess to avoid cross-library
threading conflicts that previously caused crashes. By default the script forces
headless, CPU-only execution. Developers can opt into the old in-process mode
with ``--inprocess``.
"""

import os
import argparse
import glob
import subprocess
import sys
import shlex

# Parent process hardening
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("SMOKE_MODE", "1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("indir")
    p.add_argument("--pattern", default="*.csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--inprocess", action="store_true",
                   help="Run entire batch in one process (may crash)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    assert files, f"No files matched {args.pattern} in {args.indir}"

    os.makedirs(args.outdir, exist_ok=True)

    if args.inprocess:
        cmd = [
            sys.executable,
            "-m",
            "tools._run_one_batch",
            "--indir",
            args.indir,
            "--pattern",
            args.pattern,
            "--outdir",
            args.outdir,
            "--seed",
            str(args.seed),
            "--filename",
            os.path.basename(files[0]),
        ]
        subprocess.check_call(cmd)
    else:
        for f in files:
            cmd = [
                sys.executable,
                "-m",
                "tools._run_one_batch",
                "--indir",
                args.indir,
                "--pattern",
                os.path.basename(f),
                "--outdir",
                args.outdir,
                "--seed",
                str(args.seed),
                "--filename",
                os.path.basename(f),
            ]
            print("Running:", " ".join(shlex.quote(c) for c in cmd))
            subprocess.check_call(cmd)

    ok = False
    for f in files:
        base = os.path.join(
            args.outdir, os.path.splitext(os.path.basename(f))[0]
        )
        for suf in ("_fit.csv", "_trace.csv", "_uncertainty.csv", "_uncertainty.txt"):
            path = base + suf
            assert os.path.exists(path), f"Missing {path}"
        ok = True
    assert ok
    print("OK: batch outputs present in", args.outdir)


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    main()

