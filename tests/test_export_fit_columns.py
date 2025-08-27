import os
import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak, pseudo_voigt, pseudo_voigt_area  # noqa: E402
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def test_export_fit_columns(tmp_path, monkeypatch):
    root = tk.Tk(); root.withdraw()
    app = PeakFitApp(root)
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    app.x = x
    app.y_raw = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    app.peaks = [peak]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]
    app.fit_xmax = x[-1]
    app._run_asymptotic_uncertainty()

    out = tmp_path / "fit.csv"
    app.file_label = tk.Label(root, text="sample")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)

    app.on_export()

    df = pd.read_csv(out)
    cols = [
        "solver_choice",
        "solver_loss",
        "solver_weight",
        "solver_fscale",
        "solver_maxfev",
        "solver_restarts",
        "solver_jitter_pct",
        "use_baseline",
        "baseline_mode",
        "baseline_uses_fit_range",
        "als_niter",
        "als_thresh",
        "perf_numba",
        "perf_gpu",
        "perf_cache_baseline",
        "perf_seed_all",
        "perf_max_workers",
        "bounds_center_lo",
        "bounds_center_hi",
        "bounds_fwhm_lo",
        "bounds_height_lo",
        "bounds_height_hi",
        "x_scale",
    ]
    for c in cols:
        assert c in df.columns
    root.destroy()

