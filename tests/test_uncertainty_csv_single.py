import os
import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak, pseudo_voigt
from core import data_io

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)


def test_uncertainty_csv_single(tmp_path):
    tk = pytest.importorskip("tkinter")
    root = tk.Tk()
    root.withdraw()
    app = PeakFitApp(root)
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    app.x = x
    app.y_raw = y
    app.peaks = [peak]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]
    app.fit_xmax = x[-1]
    app._run_asymptotic_uncertainty()
    paths = data_io.derive_export_paths(str(tmp_path / "single.csv"))
    app._maybe_export_uncertainty(
        pathlib.Path(paths["unc_txt"]),
        pathlib.Path(paths["unc_csv"]),
        pathlib.Path(paths["unc_band"]),
        0.0,
    )
    assert pathlib.Path(paths["unc_txt"]).exists()
    assert pathlib.Path(paths["unc_csv"]).exists()
    df = pd.read_csv(paths["unc_csv"])
    assert set(df.columns) == {
        "file",
        "peak",
        "param",
        "value",
        "stderr",
        "ci_lo",
        "ci_hi",
        "method",
        "rmse",
        "dof",
    }
    assert {"center", "height", "fwhm"}.issubset(set(df["param"]))
    root.destroy()
