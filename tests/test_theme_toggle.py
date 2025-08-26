import pytest

tk = pytest.importorskip("tkinter")
ttk = pytest.importorskip("tkinter.ttk")

from ui.app import PeakFitApp


def test_theme_toggle_changes_facecolors_and_treeview():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    style = ttk.Style()
    app.apply_theme("Dark")
    dark_fc = app.ax.get_facecolor()[:3]
    dark_tree_fg = style.lookup("Treeview", "foreground")
    assert dark_fc != (1.0, 1.0, 1.0)
    app.apply_theme("Light")
    rgb = app.ax.get_facecolor()
    assert all(abs(ch - 1.0) < 1e-6 for ch in rgb[:3])
    light_tree_fg = style.lookup("Treeview", "foreground")
    assert dark_tree_fg != light_tree_fg
    root.destroy()
