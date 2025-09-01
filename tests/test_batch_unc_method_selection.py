import types
from pathlib import Path
import numpy as np

from batch.runner import run_batch

def _fake_fit_ctx(x, y):
    peak = types.SimpleNamespace(
        center=1.0,
        height=1.0,
        fwhm=1.0,
        eta=0.5,
        lock_center=False,
        lock_width=False,
    )
    return {
        "theta": np.zeros(4),
        "residual_fn": lambda th: np.array([0.0]),
        "jacobian": np.zeros((1, 4)),
        "ymodel_fn": lambda th: np.array([0.0]),
        "peaks_out": [peak],
        "fit_ok": True,
        "rmse": 1.0,
        "dof": 1,
    }

def test_batch_uses_bayesian_when_configured(monkeypatch, tmp_path):
    f = tmp_path / "toy.txt"
    f.write_text("0 0\n1 1\n2 0\n")

    from core import data_io as dio
    monkeypatch.setattr(dio, "load_xy", lambda p: (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])))

    from core import signals
    monkeypatch.setattr(signals, "als_baseline", lambda y, **k: np.zeros_like(y))

    from core import fit_api
    def _fake_run_fit_consistent(x, y, peaks_in, config, baseline, mode, mask, **kwargs):
        return _fake_fit_ctx(x, y)
    monkeypatch.setattr(fit_api, "run_fit_consistent", _fake_run_fit_consistent)

    called = {"asym": False, "bayes": False, "boot": False}
    from core import uncertainty as unc
    monkeypatch.setattr(unc, "asymptotic_ci", lambda *a, **k: (called.__setitem__("asym", True) or {"label": "Asymptotic"}))
    monkeypatch.setattr(unc, "bayesian_ci", lambda *a, **k: (called.__setitem__("bayes", True) or {"label": "Bayesian (MCMC)", "stats": [{"center": {"est": 1, "sd": 0.1}, "height": {"est": 1, "sd": 0.1}, "fwhm": {"est": 1, "sd": 0.1}, "eta": {"est": 0.5, "sd": 0.1}}]}))
    monkeypatch.setattr(unc, "bootstrap_ci", lambda *a, **k: (called.__setitem__("boot", True) or {"label": "Bootstrap"}))

    out_dir = tmp_path / "out"
    cfg = {
        "output_dir": str(out_dir),
        "peaks": [{"center": 1.0, "height": 1.0, "fwhm": 1.0, "eta": 0.5}],
        "solver": "classic",
        "unc_method": "bayesian",
    }

    ok, processed = run_batch([str(f)], cfg, compute_uncertainty=True, unc_method=None, log=lambda m: None)
    assert ok == 1 and processed == 1
    assert called["bayes"] is True
    assert called["asym"] is False

    per_file_unc = out_dir / (f.stem + "_uncertainty.csv")
    assert per_file_unc.exists()
    assert "Bayesian (MCMC)" in per_file_unc.read_text()
