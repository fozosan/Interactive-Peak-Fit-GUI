import os
import numpy as np
import pandas as pd
import pytest
from tkinter import filedialog, messagebox

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")


def _run_once(tmp_path, write_wide):
    from ui.app import PeakFitApp, Peak, pseudo_voigt

    root = tk.Tk()
    app = PeakFitApp(root)
    app.cfg["export_unc_wide"] = bool(write_wide)

    x = np.linspace(-3, 3, 201)
    pk = Peak(0.0, 1.0, 0.8, 0.3)
    app.x = x
    app.y_raw = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
    app.peaks = [pk]
    app.use_baseline.set(False)
    try:
        app.unc_method.set("Asymptotic")
    except Exception:
        pass

    out = tmp_path / ("w_true.csv" if write_wide else "w_false.csv")
    monkey = filedialog
    monkey.asksaveasfilename = lambda **k: str(out)
    messagebox.showinfo = lambda *a, **k: None
    app.on_export()
    root.destroy()
    base = out.with_suffix("")
    wide = base.with_name(base.stem + "_uncertainty_wide.csv")
    return wide.exists()


def test_wide_toggle(tmp_path):
    assert _run_once(tmp_path, True) is True
    assert _run_once(tmp_path, False) is False

