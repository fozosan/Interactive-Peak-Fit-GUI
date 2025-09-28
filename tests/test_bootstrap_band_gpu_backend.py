import importlib.util
import numpy as np
import pytest

from core.uncertainty import bootstrap_ci

cp_spec = importlib.util.find_spec("cupy")


@pytest.mark.skipif(cp_spec is None, reason="CuPy not installed")
def test_band_backend_cupy_when_enabled(monkeypatch):
    import importlib

    cp = importlib.import_module("cupy")

    x = np.linspace(0, 1, 128)
    theta = np.array([0.3, 1.0, 0.2, 0.5])
    rng = np.random.default_rng(0)
    y_cpu = np.sin(2 * np.pi * x) + 0.01 * rng.normal(size=x.size)
    resid = y_cpu - y_cpu.mean()
    J = np.ones((x.size, theta.size))

    def refit(th0, locked_mask, bounds, x_in, y_in):
        return np.asarray(th0, float), True

    def predict_full(th):
        return cp.asarray(np.sin(2 * np.pi * x))

    fit_ctx = {
        "x_all": x,
        "y_all": y_cpu,
        "baseline": None,
        "mode": "add",
        "refit": refit,
        "strict_refit": True,
        "bootstrap_jitter": 0.05,
        "unc_use_gpu": True,
        "unc_band_workers": 0,
        "peaks": [object()],
    }
    monkeypatch.setenv("PEAKFIT_USE_GPU", "1")

    res = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=predict_full,
        x_all=x,
        y_all=y_cpu,
        fit_ctx=fit_ctx,
        n_boot=64,
        seed=123,
        workers=0,
        alpha=0.1,
        center_residuals=True,
        return_band=True,
    )
    assert res.band is not None
    assert res.diagnostics.get("band_backend") == "cupy"
