import os
import numpy as np
import pytest

from core.peaks import Peak
from infra import performance
from uncertainty import bootstrap


def _residual_builder_factory(x, y):
    def _residual(theta):
        a = float(theta[0])
        b = float(theta[1])
        yhat = a * x + b
        return yhat - y

    return _residual


@pytest.mark.parametrize(
    "strategy,threads,expect",
    [
        ("outer", 0, "1"),
        ("inner", 3, "3"),
    ],
)
def test_blas_env_clamp_pass_through(strategy, threads, expect, monkeypatch):
    pytest.importorskip("threadpoolctl")

    x = np.linspace(0.0, 1.0, 32)
    a_true, b_true = 0.75, 0.1
    y = a_true * x + b_true
    theta = np.array([a_true, b_true, 0.05, 0.15], dtype=float)

    template_peak = Peak(
        center=float(theta[0]),
        height=float(theta[1]),
        fwhm=float(theta[2]),
        eta=float(theta[3]),
        lock_center=False,
        lock_width=False,
    )

    def _fake_solve(x_arr, y_boot, peaks, mode, baseline, options):
        return {"theta": np.array(theta, dtype=float)}

    from fit import classic as _classic

    monkeypatch.setattr(_classic, "solve", _fake_solve)

    captured = {"mkl": None, "openblas": None, "omp": None}
    orig_worker = bootstrap._bootstrap_worker

    def _capture_worker(args):
        captured["mkl"] = os.environ.get("MKL_NUM_THREADS")
        captured["openblas"] = os.environ.get("OPENBLAS_NUM_THREADS")
        captured["omp"] = os.environ.get("OMP_NUM_THREADS")
        return orig_worker(args)

    monkeypatch.setattr(bootstrap, "_bootstrap_worker", _capture_worker)

    for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)

    resample_cfg = {
        "x": x,
        "y": y,
        "peaks": [template_peak],
        "mode": "add",
        "baseline": None,
        "theta": theta,
        "options": {},
        "n": 2,
        "seed": 21,
        "workers": 1,
        "perf_parallel_strategy": strategy,
        "perf_blas_threads": threads,
    }

    residual_builder = _residual_builder_factory(x, y)

    performance.apply_global_seed(None, False)
    bootstrap.bootstrap("classic", resample_cfg, residual_builder)

    assert captured["mkl"] == expect
    assert captured["openblas"] == expect
    assert captured["omp"] == expect
