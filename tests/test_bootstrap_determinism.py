import numpy as np

from infra import performance
from uncertainty import bootstrap
from core.peaks import Peak


def _residual_builder_factory(x, y):
    def _residual(theta):
        a = float(theta[0])
        b = float(theta[1])
        yhat = a * x + b
        return yhat - y

    return _residual


def test_bootstrap_determinism_seed_all_true(monkeypatch):
    x = np.linspace(0.0, 1.0, 64)
    a_true, b_true = 1.5, -0.25
    y = a_true * x + b_true

    theta = np.array([a_true, b_true, 0.1, 0.2], dtype=float)

    # Minimal peak template so bootstrap worker can build Peak copies
    template_peak = Peak(
        center=float(theta[0]),
        height=float(theta[1]),
        fwhm=float(theta[2]),
        eta=float(theta[3]),
        lock_center=False,
        lock_width=False,
    )

    def _fake_solve(x_arr, y_boot, peaks, mode, baseline, options):
        base = np.array(theta, dtype=float)
        noise = np.random.normal(scale=1e-3, size=base.shape)
        return {"theta": base + noise}

    from fit import classic as _classic

    monkeypatch.setattr(_classic, "solve", _fake_solve)

    resample_cfg = {
        "x": x,
        "y": y,
        "peaks": [template_peak],
        "mode": "add",
        "baseline": None,
        "theta": theta,
        "options": {},
        "n": 10,
        "seed": 123,
        "workers": 1,
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
    }

    residual_builder = _residual_builder_factory(x, y)

    performance.apply_global_seed(999, True)
    out1 = bootstrap.bootstrap("classic", resample_cfg, residual_builder)
    performance.apply_global_seed(999, True)
    out2 = bootstrap.bootstrap("classic", resample_cfg, residual_builder)

    t1 = np.asarray(out1["params"]["theta"], dtype=float)
    t2 = np.asarray(out2["params"]["theta"], dtype=float)
    c1 = np.asarray(out1["params"]["cov"], dtype=float)
    c2 = np.asarray(out2["params"]["cov"], dtype=float)

    assert np.allclose(t1, t2)
    assert np.allclose(c1, c2)
