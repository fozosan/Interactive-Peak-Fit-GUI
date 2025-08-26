import os, numpy as np, pytest
from matplotlib.collections import PolyCollection
headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")

from ui.app import PeakFitApp, Peak, pseudo_voigt


def test_ci_band_computed_and_plotted():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    x = np.linspace(-5, 5, 101)
    peak = Peak(center=0.0, height=1.0, fwhm=1.0, eta=0.5)
    app.x = x
    app.y_raw = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    app.peaks = [peak]
    app.use_baseline.set(False)
    app.fit_xmin = x[0]
    app.fit_xmax = x[-1]
    app._run_asymptotic_uncertainty()
    assert app.ci_band is not None
    app.show_ci_band = True
    app.refresh_plot()
    bands = [c for c in app.ax.collections if isinstance(c, PolyCollection)]
    assert bands
    root.destroy()
