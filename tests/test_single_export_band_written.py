import os, pathlib, numpy as np, pandas as pd, pytest
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY","")=="" and os.name!="nt")
if headless: pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")

def test_single_export_band_written(tmp_path, monkeypatch):
    from ui.app import PeakFitApp, Peak, pseudo_voigt
    root = tk.Tk()
    app = PeakFitApp(root)

    x = np.linspace(-3,3,241)
    pk = Peak(0.0, 1.0, 0.8, 0.3)
    app.x = x
    app.y_raw = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
    app.peaks = [pk]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]; app.fit_xmax = x[-1]

    # Force Asymptotic in UI
    try:
        app.unc_method.set("Asymptotic")
    except Exception:
        pass

    out = tmp_path / "s.csv"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)

    app.on_export()

    assert (tmp_path / "s_uncertainty.csv").exists()
    # Asymptotic should write band
    assert (tmp_path / "s_uncertainty_band.csv").exists()
    root.destroy()
