"""Template auto-apply behaviour in GUI and batch modes."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import fit_api, models, peaks, signals
from batch import runner


headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
skip_if_no_display = pytest.mark.skipif(headless, reason="requires display")


def test_batch_template_auto_apply(tmp_path):
    x = np.linspace(-5, 5, 101)
    pk = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    y = models.pv_sum(x, [pk])
    fpath = tmp_path / "spec.csv"
    np.savetxt(fpath, np.column_stack([x, y]), delimiter=",")

    cfg_single = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "perf_seed_all": True,
    }
    baseline = signals.als_baseline(y, lam=1e5, p=0.001, niter=10, tol=0.0)
    res = fit_api.run_fit_consistent(
        x,
        y,
        [pk],
        cfg_single,
        baseline,
        "add",
        np.ones_like(x, bool),
        rng_seed=123,
    )

    cfg_batch = {
        "peaks": [pk.__dict__],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "source": "template",
        "output_dir": str(tmp_path),
        "output_base": "batch",
    }
    runner.run([str(fpath)], cfg_batch)
    df = pd.read_csv(tmp_path / "batch_fit.csv")

    cols = ["center", "height", "fwhm", "eta"]
    for i, c in enumerate(cols):
        assert pytest.approx(df[c].iloc[0], rel=1e-6) == res["theta"][i]


@skip_if_no_display
def test_gui_template_auto_apply(tmp_path, monkeypatch):
    from ui import app as app_module
    tk = pytest.importorskip("tkinter")

    pk = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    template = {"templates": {"t1": [pk.__dict__]}, "auto_apply_template": True, "auto_apply_template_name": "t1"}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(template))
    monkeypatch.setattr(app_module, "CONFIG_PATH", cfg_path)

    x = np.linspace(-5, 5, 101)
    y = models.pv_sum(x, [pk])
    data_path = tmp_path / "d.csv"
    np.savetxt(data_path, np.column_stack([x, y]), delimiter=",")

    root = tk.Tk(); root.withdraw()
    app = app_module.PeakFitApp(root)
    # simulate file load (inner logic of open_file)
    x2, y2 = app_module.load_xy_any(str(data_path))
    app.x, app.y_raw = x2, y2
    app.compute_baseline()
    app.peaks.clear()
    if app.auto_apply_template.get():
        t = app._templates()
        name = app.auto_apply_template_name.get()
        if name in t:
            app._apply_template_list(t[name], reheight=True)

    assert len(app.peaks) == 1
    assert pytest.approx(app.peaks[0].center, rel=1e-6) == pk.center
    root.destroy()

