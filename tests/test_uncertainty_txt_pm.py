import os
import sys
import pathlib
import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak, pseudo_voigt  # noqa: E402
from core import data_io  # noqa: E402

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)


def test_uncertainty_txt_pm(tmp_path):
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
    with open(paths["unc_txt"], encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    idx = lines.index("Peak 1")
    section = lines[idx + 1 : idx + 5]
    for pname in ("center", "height", "fwhm", "eta"):
        line = next(l for l in section if l.strip().startswith(pname))
        assert " ± " in line or "(fixed)" in line
        assert "95% CI" in line
    root.destroy()
