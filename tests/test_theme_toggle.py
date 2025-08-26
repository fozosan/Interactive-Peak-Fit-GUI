import pytest

tk = pytest.importorskip("tkinter")

from ui.app import PeakFitApp


def test_theme_toggle_changes_facecolors():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    app.theme_var.set("Dark")
    app.apply_theme()
    assert tuple(app.fig.get_facecolor()) != (1.0, 1.0, 1.0, 1.0)
    app.theme_var.set("Light")
    app.apply_theme()
    assert tuple(app.fig.get_facecolor()) == (1.0, 1.0, 1.0, 1.0)
    root.destroy()
