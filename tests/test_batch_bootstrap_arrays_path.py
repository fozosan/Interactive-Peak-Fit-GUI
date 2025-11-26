import numpy as np
from pathlib import Path

from tests.conftest import bayes_knobs, bootstrap_cfg, ensure_unc_common

def test_batch_bootstrap_runs_with_arrays(tmp_path, monkeypatch):
    f = tmp_path / "toy.txt"
    f.write_text("0 0\n1 1\n2 0\n")

    from core import data_io as dio
    monkeypatch.setattr(dio, "load_xy",
        lambda p: (np.array([0.0,1.0,2.0]), np.array([0.0,1.0,0.0])))
    from core import signals
    monkeypatch.setattr(signals, "als_baseline", lambda y, **k: np.zeros_like(y))
    from core import fit_api
    monkeypatch.setattr(fit_api, "run_fit_consistent", lambda *a, **k: {
        "fit_ok": True,
        "theta": np.array([1.0, 1.0, 1.0, 0.5]),
        "x": a[0], "y": a[1], "x_fit": a[0].tolist(), "y_fit": a[1].tolist(),
        "peaks": [],
        "peaks_out": [type("P", (), {"center":1.0,"height":1.0,"fwhm":1.0,"eta":0.5,"lock_center":False,"lock_width":False})()],
        "baseline": np.zeros_like(a[0]), "mode": "add",
        "rmse": 1.0,
        "residual_fn": lambda th: a[1] - 0*a[0],
        "jacobian": None,
        "predict_full": lambda th, x=a[0]: np.zeros_like(x),
        "dof": max(1, a[0].size - 4),
        "solver": "classic",
    })

    from batch.runner import run_batch
    out = tmp_path / "out"
    cfg = {
        "output_dir": str(out),
        "peaks": [{"center":1.0,"height":1.0,"fwhm":1.0,"eta":0.5}],
        "solver": "classic",
        "unc_method": "bootstrap",
    }
    cfg.update(ensure_unc_common({}))
    cfg.update(bootstrap_cfg(n=48))
    cfg.update(bayes_knobs())
    n_ok, total = run_batch([str(f)], cfg, compute_uncertainty=True)
    assert n_ok == total == 1
    stem = Path(f).stem
    assert (out / f"{stem}_uncertainty.txt").exists()
