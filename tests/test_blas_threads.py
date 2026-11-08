"""BLAS thread pools: one thread from startup, and scoped limits that undo themselves.

OpenBLAS starts one thread per CPU. Every array in this app is small, so the
pool only costs time: a 15-file batch took about 20 s at 24 threads and 0.2 s
at one. The pool is sized when NumPy loads, so the entry points set the thread
variables before that. Separately, ``blas_limit_ctx`` and
``blas_single_thread_ctx`` left the pool clamped after they exited, and turned
any exception raised inside them into "generator didn't stop after throw()".
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

from infra import performance  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

SCOPED_LIMITS = [
    pytest.param(lambda: performance.blas_limit_ctx(1), id="blas_limit_ctx(1)"),
    pytest.param(performance.blas_single_thread_ctx, id="blas_single_thread_ctx()"),
]


def _blas_threads():
    sizes = sorted({d["num_threads"] for d in threadpool_info() if d["user_api"] == "blas"})
    if not sizes:
        pytest.skip("no BLAS library visible to threadpoolctl")
    return sizes


@pytest.mark.parametrize("scoped_limit", SCOPED_LIMITS)
def test_scoped_limit_is_undone_on_exit(scoped_limit):
    # Start from a known pool size, so a leak shows whatever this process began with.
    with threadpool_limits(limits=3):
        assert _blas_threads() == [3]
        with scoped_limit():
            assert _blas_threads() == [1]
        assert _blas_threads() == [3]


@pytest.mark.parametrize("scoped_limit", SCOPED_LIMITS)
def test_exception_inside_scoped_limit_propagates(scoped_limit):
    with threadpool_limits(limits=3):
        with pytest.raises(ValueError, match="boom from the fit"):
            with scoped_limit():
                raise ValueError("boom from the fit")
        assert _blas_threads() == [3]


@pytest.mark.parametrize("threads", [None, 0])
def test_no_limit_leaves_the_pool_alone(threads):
    with threadpool_limits(limits=3):
        with performance.blas_limit_ctx(threads):
            assert _blas_threads() == [3]
        assert _blas_threads() == [3]


@pytest.mark.parametrize("entry_point", ["run_app", "tools.batch_cli"])
@pytest.mark.parametrize("preset, expected", [(None, "[1]"), ("2", "[2]")])
def test_entry_point_sizes_the_blas_pool_before_numpy_loads(entry_point, preset, expected, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    if preset is not None:
        # A thread count the user set in the environment is kept. OpenBLAS caps
        # its starting pool at the CPU count.
        if (os.cpu_count() or 1) < int(preset):
            pytest.skip("needs as many CPUs as the preset thread count")
        env["OPENBLAS_NUM_THREADS"] = preset
    env["HOME"] = env["USERPROFILE"] = str(tmp_path)
    env["MPLBACKEND"] = "Agg"
    code = (
        f"import {entry_point}\n"
        "from threadpoolctl import threadpool_info\n"
        "print(sorted({d['num_threads'] for d in threadpool_info() if d['user_api'] == 'blas'}))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected
