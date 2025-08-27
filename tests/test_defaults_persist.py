"""Verify default config values and persistence of toggles."""

import os
import sys
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui import app as app_module  # noqa: E402

headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def test_defaults_persist(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    monkeypatch.setattr(app_module, "CONFIG_PATH", cfg_path)

    root = tk.Tk(); root.withdraw()
    app = app_module.PeakFitApp(root)
    assert app.baseline_use_range.get() is True
    assert app.show_ci_band_var.get() is True
    app.baseline_use_range.set(False)
    app.show_ci_band_var.set(False)
    app.perf_numba.set(True)
    app.perf_gpu.set(True)
    root.destroy()

    root2 = tk.Tk(); root2.withdraw()
    app2 = app_module.PeakFitApp(root2)
    assert app2.baseline_use_range.get() is False
    assert app2.show_ci_band_var.get() is False
    cfg = app_module.load_config()
    assert cfg["perf_numba"] is True
    assert cfg["perf_gpu"] is True
    root2.destroy()

