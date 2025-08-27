import os
import sys
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui.app import PeakFitApp  # noqa: E402

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)


def test_unc_workers_vs_seed_all(monkeypatch):
    tk = pytest.importorskip("tkinter")
    root = tk.Tk()
    root.withdraw()
    app = PeakFitApp(root)
    label_text = app.unc_workers_label.cget("text")
    app.perf_seed_all.set(True)
    root.update_idletasks()
    assert app.unc_workers_label.cget("text") == label_text
    app.unc_workers_var.set(2)
    assert app._resolve_unc_workers() == 2
    app.unc_workers_var.set(0)
    app.perf_max_workers.set(3)
    assert app._resolve_unc_workers() == 3
    app.perf_max_workers.set(0)
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    assert app._resolve_unc_workers() == 5
    root.destroy()
