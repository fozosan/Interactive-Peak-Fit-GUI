import os
import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui import app as app_module  # noqa: E402
from batch import runner  # noqa: E402
from ui.app import Peak, pseudo_voigt  # noqa: E402

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)


def test_templates_auto_apply(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    cfg = {
        "templates": {"t1": [peak.__dict__]},
        "auto_apply_template": True,
        "auto_apply_template_name": "t1",
    }
    root = tk.Tk(); root.withdraw()
    app = app_module.PeakFitApp(root, cfg=cfg)
    x = np.linspace(-5, 5, 101)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    data_file = tmp_path / "single.csv"
    np.savetxt(data_file, np.column_stack([x, y]), delimiter=",")
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: str(data_file))
    app.on_open()
    assert len(app.peaks) == 1
    assert app.peaks[0].center == pytest.approx(0.0)
    root.destroy()

    # Batch: template applied to each file
    for i in range(2):
        np.savetxt(tmp_path / f"b{i}.csv", np.column_stack([x, y]), delimiter=",")
    cfg_batch = {
        "peaks": [peak.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        "classic": {},
        "baseline_uses_fit_range": True,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
        "output_dir": str(tmp_path),
        "output_base": "batch",
    }
    runner.run([str(tmp_path / "b*.csv")], cfg_batch)
    fit_df = pd.read_csv(tmp_path / "batch_fit.csv")
    assert len(fit_df) == 2
