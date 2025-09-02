import os, sys, pathlib, numpy as np, pandas as pd, pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak, pseudo_voigt
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY","" )=="" and os.name!="nt")
if headless: pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")


def test_single_export_always_computes_uncertainty(tmp_path, monkeypatch):
    root = tk.Tk()
    app = PeakFitApp(root)

    # Simulate batch toggle OFF to prove single export ignores it
    try:
        app.batch_unc_enabled.set(False)
    except Exception:
        pass
    app.cfg["batch_compute_uncertainty"] = False
    app.cfg["compute_uncertainty_batch"] = False

    x = np.linspace(-5,5,201)
    pk = Peak(0.0, 1.0, 1.0, 0.5)
    app.x = x
    app.y_raw = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
    app.peaks = [pk]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]; app.fit_xmax = x[-1]
    app.fit()

    out = tmp_path / "single.csv"
    app.file_label = tk.Label(root, text="sample")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)

    app.on_export()

    # Must exist even when batch toggle is OFF
    unc = pd.read_csv(tmp_path / "single_uncertainty.csv")
    assert {"file","peak","param","value","stderr","method","rmse","dof"}.issubset(set(unc.columns))
    root.destroy()
