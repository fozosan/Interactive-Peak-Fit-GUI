import os
import numpy as np
import pandas as pd
import pytest
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")


def test_single_export_ignores_history_and_uses_ui_method(tmp_path, monkeypatch):
    from ui.app import PeakFitApp, Peak, pseudo_voigt

    root = tk.Tk()
    app = PeakFitApp(root)

    x = np.linspace(-4, 4, 321)
    pk = Peak(0.0, 1.2, 0.9, 0.3)
    app.x = x
    app.y_raw = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
    app.peaks = [pk]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]
    app.fit_xmax = x[-1]

    # Set a fake prior uncertainty result to verify it is ignored
    app.last_uncertainty = {
        "label": "Bootstrap",
        "stats": [{"peak": 1, "param": "center", "value": pk.center, "stderr": 0.1}],
    }

    # Force Asymptotic in UI so export must recompute with this method
    try:
        app.unc_method.set("Asymptotic")
    except Exception:
        pass

    out = tmp_path / "e.csv"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)

    app.on_export()

    df = pd.read_csv(tmp_path / "e_uncertainty.csv")
    methods = set(df["method"].astype(str).str.lower())
    assert "asymptotic" in methods and "bootstrap" not in methods

    root.destroy()

