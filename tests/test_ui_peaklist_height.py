import os, sys, pytest
headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp, Peak


def test_tree_height_matches_peaks():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    app.peaks = [Peak(center=float(i)) for i in range(12)]
    app.refresh_tree()
    assert int(app.tree['height']) == max(6, len(app.peaks))
    root.destroy()
