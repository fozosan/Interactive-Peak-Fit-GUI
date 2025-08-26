import os, warnings, numpy as np, pytest
headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
from ui.app import PeakFitApp

tk = pytest.importorskip("tkinter")


def test_asymptotic_no_runtime_warning():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    app.x = np.linspace(0, 10, 200)
    app.y_raw = np.exp(-((app.x - 5) ** 2) / 2)
    app.baseline = np.zeros_like(app.x)
    app.peaks = [app.Peak(center=5.0, height=1.0, fwhm=2.0, eta=0.5)]
    app.use_baseline.set(False)
    app.baseline_mode.set("add")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", RuntimeWarning)
        app._run_asymptotic_uncertainty()
    assert app.ci_band is not None
    root.destroy()
