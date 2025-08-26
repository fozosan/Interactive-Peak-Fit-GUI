import pytest
import tkinter as tk
from ui.app import PeakFitApp, Peak


@pytest.mark.skipif(
    "DISPLAY" not in __import__("os").environ and __import__("platform").system() != "Windows",
    reason="GUI test requires a display or Windows Tk",
)
def test_legend_toggle_smoke():
    root = tk.Tk()
    app = PeakFitApp(root)

    import numpy as np

    app.x = np.linspace(0, 10, 200)
    app.y_raw = np.sin(app.x) + 2.0
    app.baseline = np.zeros_like(app.y_raw)

    app.peaks = [Peak(center=5.0, height=1.0, fwhm=1.0, eta=0.5)]
    app.components_visible = True

    app.show_legend_var.set(True)
    app.refresh_plot()
    assert app.ax.get_legend() is not None

    app.show_legend_var.set(False)
    app.refresh_plot()
    assert app.ax.get_legend() is None

    app.show_legend_var.set(True)
    app.refresh_plot()
    assert app.ax.get_legend() is not None

    root.destroy()

