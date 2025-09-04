import os
import sys
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui import app  # noqa: E402

headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def test_baseline_method_persistence(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    monkeypatch.setattr(app, "CONFIG_PATH", cfg_path)

    root1 = tk.Tk()
    root1.withdraw()
    a1 = app.PeakFitApp(root1)
    a1.base_method_var.set("polynomial")
    a1.poly_degree_var.set(4)
    a1.poly_norm_var.set(False)
    a1.save_baseline_default()
    root1.destroy()

    root2 = tk.Tk()
    root2.withdraw()
    a2 = app.PeakFitApp(root2)
    assert a2.base_method_var.get() == "polynomial"
    assert a2.poly_degree_var.get() == 4
    assert bool(a2.poly_norm_var.get()) is False
    root2.destroy()
