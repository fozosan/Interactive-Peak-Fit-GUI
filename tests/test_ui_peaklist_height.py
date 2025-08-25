import pytest
import pathlib, sys

tk = pytest.importorskip("tkinter")

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
