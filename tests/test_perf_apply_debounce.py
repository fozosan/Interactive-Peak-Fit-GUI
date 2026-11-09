"""Performance settings: what a change saves, applies and logs.

A variable trace asks for an apply through request_apply_performance(), which
runs it 150 ms later, so a burst of writes becomes one apply and one log line.
The setting itself must reach the config file at once: a window closed inside
those 150 ms would otherwise lose it.
"""
import json
import os
import time

import pytest

from infra import performance
from ui import app as app_module

headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


@pytest.fixture
def gui(monkeypatch):
    """A PeakFitApp, the configs it saves and the log lines it writes after startup."""
    saves = []
    real_save = app_module.save_config

    def counting_save(cfg):
        saves.append(dict(cfg))
        return real_save(cfg)

    monkeypatch.setattr(app_module, "save_config", counting_save)
    root = tk.Tk()
    root.withdraw()
    app = app_module.PeakFitApp(root)
    lines = []
    monkeypatch.setattr(app, "log", lambda msg, *a, **k: lines.append(str(msg)))
    try:
        yield app, saves, lines
    finally:
        root.destroy()
        performance.set_numba(False)
        performance.set_gpu(False)


def _backend_lines(lines):
    return [line for line in lines if line.startswith("Backend:")]


def _run_events(root, until, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not until():
        root.update()
        time.sleep(0.01)
    # let anything still scheduled inside the debounce window run too
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)


def test_a_change_reaches_the_config_before_the_debounced_apply(gui):
    app, saves, lines = gui
    app.perf_unc_workers.set(5)
    assert app._perf_apply_after_id is not None  # the apply is still pending
    assert json.loads(app_module.CONFIG_PATH.read_text())["perf_unc_workers"] == 5


def test_a_burst_of_writes_is_applied_and_logged_once(gui):
    app, saves, lines = gui
    for workers in range(1, 11):
        app.perf_unc_workers.set(workers)
    _run_events(app.root, until=lambda: _backend_lines(lines))
    assert len(_backend_lines(lines)) == 1
    assert "unc_workers_req=10" in _backend_lines(lines)[0]


def test_applying_the_same_settings_again_logs_nothing(gui):
    app, saves, lines = gui
    for _ in range(3):
        app.request_apply_performance(source="test", immediate=True)
    assert _backend_lines(lines) == []


def test_the_apply_button_always_logs(gui):
    app, saves, lines = gui
    app._apply_performance_immediate()
    app._apply_performance_immediate()
    assert len(_backend_lines(lines)) == 2


def test_turning_numba_on_logs_the_new_backend(gui):
    if not performance._NUMBA_OK:
        pytest.skip("numba not installed")
    app, saves, lines = gui
    app.perf_numba.set(True)
    _run_events(app.root, until=lambda: _backend_lines(lines))
    assert [line.split(" |")[0] for line in _backend_lines(lines)] == ["Backend: numba"]


def test_save_cfg_false_does_not_write_the_config(gui):
    app, saves, lines = gui
    before = len(saves)
    app.request_apply_performance(source="test", immediate=True, save_cfg=False)
    assert len(saves) == before
