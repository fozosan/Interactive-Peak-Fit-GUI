import types
import pytest
import numpy as np
from pathlib import Path
from batch.runner import run_batch
from tests.conftest import bayes_knobs, bootstrap_cfg, ensure_unc_common


def _fake_fit_ctx(x, y):
    peak = types.SimpleNamespace(center=1.0, height=1.0, fwhm=1.0, eta=0.5, lock_center=False, lock_width=False)
    return {
        "fit_ok": True,
        "theta": np.array([1.0, 1.0, 1.0, 0.5]),
        "x": x,
        "y": y,
        "x_fit": list(x),
        "y_fit": list(y),
        "peaks": [peak],
        "peaks_out": [peak],
        "baseline": np.zeros_like(x),
        "mode": "add",
        "residual_fn": lambda th: y - 0.0 * x,
        "jacobian": np.eye(4),
        "predict_full": lambda th, x=x: np.zeros_like(x),
        "rmse": 0.0,
        "dof": max(1, x.size - 4),
    }


def test_unknown_unc_method_raises(tmp_path, monkeypatch):
    f = tmp_path / "toy.txt"
    f.write_text("0 0\n1 1\n2 0\n")

    from core import data_io as dio
    monkeypatch.setattr(dio, "load_xy", lambda p: (np.array([0.0,1.0,2.0]), np.array([0.0,1.0,0.0])))

    from core import signals
    monkeypatch.setattr(signals, "als_baseline", lambda y, **k: np.zeros_like(y))

    from core import fit_api
    monkeypatch.setattr(fit_api, "run_fit_consistent", lambda *a, **k: _fake_fit_ctx(*a[:2]))

    out_dir = tmp_path / "out"
    cfg = {
        "output_dir": str(out_dir),
        "peaks": [{"center":1.0,"height":1.0,"fwhm":1.0,"eta":0.5}],
        "solver": "classic",
        "unc_method": "totally-unknown-method",
    }
    cfg.update(ensure_unc_common({}))
    cfg.update(bootstrap_cfg(n=32))
    cfg.update(bayes_knobs())

    with pytest.raises(ValueError):
        run_batch([str(f)], cfg, compute_uncertainty=True, unc_method=None, log=lambda m: None)
