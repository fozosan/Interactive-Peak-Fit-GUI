import os, sys, pytest
headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)
tk = pytest.importorskip("tkinter")
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp


def test_classic_advanced_collapsed():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("requires display")
    root.withdraw()
    app = PeakFitApp(root)
    app.solver_choice.set('classic')
    app._on_solver_change()
    assert not app.classic_adv_frame.winfo_ismapped()
    root.destroy()
