"""Convenience launcher for the Peakfit 3.x prototype UI.

Ensures that package imports resolve when executed from an IDE like
PyCharm or directly from the command line by pre-pending the project
root to ``sys.path``. This avoids ``ModuleNotFoundError`` issues for
modules such as ``fit.bounds`` when the current working directory is not
the repository root.
"""

from __future__ import annotations

import os
import sys

# One BLAS/OpenMP thread. Every array here is small, and OpenBLAS's default of
# one thread per CPU made a 15-file batch about 100x slower. The pool is sized
# when NumPy loads, so this runs before any import of it. A value already set
# in the environment wins. tools/batch_cli.py does the same.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.app import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
