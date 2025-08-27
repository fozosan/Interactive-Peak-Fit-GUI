import os
import sys
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui import app as app_module  # noqa: E402

headless = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def test_config_persistence_perf_unc(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    monkeypatch.setattr(app_module, "CONFIG_PATH", cfg_path)

    root = tk.Tk(); root.withdraw()
    app = app_module.PeakFitApp(root)
    app.perf_numba.set(True)
    app.perf_gpu.set(True)
    app.perf_cache_baseline.set(False)
    app.perf_seed_all.set(True)
    app.perf_max_workers.set(5)
    app.apply_performance()

    app.show_ci_band_var.set(False)
    app._toggle_ci_band()
    app.baseline_use_range.set(False)
    app.on_baseline_use_range_toggle()

    cfg = app_module.load_config()
    assert cfg["perf_numba"] is True
    assert cfg["perf_gpu"] is True
    assert cfg["perf_cache_baseline"] is False
    assert cfg["perf_seed_all"] is True
    assert cfg["perf_max_workers"] == 5
    assert cfg["show_uncertainty_band"] is False
    assert cfg["baseline_uses_fit_range"] is False
    root.destroy()

