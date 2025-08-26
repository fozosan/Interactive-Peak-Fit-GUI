import os, sys, types, numpy as np, pytest
headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp


def _make_app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    app.x = np.linspace(0, 10, 5)
    app.y_raw = np.zeros_like(app.x)
    app.use_baseline.set(False)
    return root, app


def test_span_preserves_toggle_off():
    root, app = _make_app()
    app.add_peaks_mode.set(False)
    init_cursor = app.canvas.get_tk_widget().cget("cursor")

    app.enable_span()
    assert app._span_active
    app._span.onselect(2.0, 3.0)

    assert not app._span_active
    assert app.add_peaks_mode.get() is False
    assert app.canvas.get_tk_widget().cget("cursor") == init_cursor
    root.destroy()


def test_span_cancel_on_leave_restores_state():
    root, app = _make_app()
    app.add_peaks_mode.set(True)
    init_cursor = app.canvas.get_tk_widget().cget("cursor")

    app.enable_span()
    assert app._span_active
    # Click should be ignored while span active
    event = types.SimpleNamespace(inaxes=app.ax, xdata=5.0, button=1)
    app.on_click_plot(event)
    assert len(app.peaks) == 0

    # Simulate leaving the figure mid-span
    app.canvas.callbacks.process("figure_leave_event", types.SimpleNamespace())

    assert not app._span_active
    assert app.add_peaks_mode.get() is True
    assert app.canvas.get_tk_widget().cget("cursor") == init_cursor
    # Click now should add a peak since toggle restored
    app.on_click_plot(event)
    assert len(app.peaks) == 1
    root.destroy()
