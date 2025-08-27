import os
import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak, pseudo_voigt  # noqa: E402
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def _sync_run(app, fn, done):
    try:
        res = fn()
        err = None
    except Exception as e:  # pragma: no cover - defensive
        res, err = None, e
    done(res, err)


def test_export_uncertainty_single(tmp_path, monkeypatch):
    root = tk.Tk(); root.withdraw()
    app = PeakFitApp(root)
    monkeypatch.setattr(app, "run_in_thread", _sync_run)
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    app.x = x
    app.y_raw = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    app.peaks = [peak]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]
    app.fit_xmax = x[-1]

    app.fit()
    assert app.ci_band is not None

    out = tmp_path / "res.csv"
    app.file_label = tk.Label(root, text="sample")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)

    app.on_export()

    unc = pd.read_csv(tmp_path / "res_uncertainty.csv")
    cols = [
        "file",
        "peak",
        "center",
        "height",
        "fwhm",
        "eta",
        "lock_center",
        "lock_width",
        "stderr_height",
        "ci95_height_lo",
        "ci95_height_hi",
        "stderr_center",
        "ci95_center_lo",
        "ci95_center_hi",
        "stderr_fwhm",
        "ci95_fwhm_lo",
        "ci95_fwhm_hi",
        "rmse",
        "dof",
        "s2",
        "method",
        "mode",
        "fit_xmin",
        "fit_xmax",
    ]
    for c in cols:
        assert c in unc.columns
    root.destroy()

