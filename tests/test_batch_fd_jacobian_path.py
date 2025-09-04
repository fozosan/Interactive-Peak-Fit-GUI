import numpy as np
import pytest
import types
from pathlib import Path


def _fake_fit_ctx(x, y):
    # One-peak theta = [c, h, fwhm, eta]
    th = np.array([1.0, 1.0, 1.0, 0.5], float)
    peak = types.SimpleNamespace(
        center=1.0, height=1.0, fwhm=1.0, eta=0.5, lock_center=False, lock_width=False
    )
    return {
        "fit_ok": True,
        "theta": th,
        "x": x,
        "y": y,
        "x_fit": list(x),
        "y_fit": list(y),
        "peaks": [peak],
        "peaks_out": [peak],
        "baseline": np.zeros_like(x),
        "mode": "add",
        # Crucially omit both 'residual_jac' and 'jacobian' to hit FD path.
        # Also omit 'residual_fn' so runner builds it via build_residual.
        "dof": max(1, x.size - 4),
        "fit_ok_msg": "ok",
        "rmse": 1.0,
    }


def test_asymptotic_uses_fd_when_no_jacobian(tmp_path, monkeypatch):
    f = tmp_path / "toy.txt"
    f.write_text("0 0\n1 1\n2 0\n")

    # Stub I/O and baseline
    from core import data_io as dio

    monkeypatch.setattr(
        dio, "load_xy", lambda p: (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])))

    from core import signals

    monkeypatch.setattr(signals, "als_baseline", lambda y, **k: np.zeros_like(y))

    # Make run_fit produce a context lacking jacobian/residual_fn
    from core import fit_api

    monkeypatch.setattr(
        fit_api, "run_fit_consistent", lambda *a, **k: _fake_fit_ctx(*a[:2])
    )

    # Execute batch with asymptotic to exercise FD path
    from batch.runner import run_batch
    from core import models

    monkeypatch.setattr(
        models, "pseudo_voigt", lambda x, h, c, fw, eta: np.zeros_like(x), raising=False
    )

    out_dir = tmp_path / "out"
    cfg = {
        "output_dir": str(out_dir),
        "peaks": [
            {"center": 1.0, "height": 1.0, "fwhm": 1.0, "eta": 0.5}
        ],
        "solver": "classic",
        "unc_method": "asymptotic",
    }

    n_ok, total = run_batch([str(f)], cfg, compute_uncertainty=True)
    n_fail = total - n_ok
    assert n_fail == 0

    # Uncertainty file should be produced
    stem = Path(f).stem
    assert (out_dir / f"{stem}_uncertainty.txt").exists()

