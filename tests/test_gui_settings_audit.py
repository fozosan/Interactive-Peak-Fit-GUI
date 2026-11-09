"""The Performance settings the GUI offers each do something (game plan 3.3).

The 2026-11-07 performance review traced every control:
- GPU switched the model to CuPy, which made fits 2.6 to 7x slower;
- GPU chunk set a value nothing reads;
- Max workers set a fit worker count nothing uses, since nothing fits in parallel.
These are gone. The rest are live: Numba, Cache baseline, Seed all and Seed,
the BLAS strategy and threads, and Unc. workers, whose "auto" (0) means
sequential: thread pools never sped the draws up, and one thread per CPU was
the slowest setting measured.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.gui_driver import Driver
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from core import uncertainty as core_uncertainty  # noqa: E402
from infra import performance  # noqa: E402
from ui import app as ui_app  # noqa: E402


@pytest.fixture
def gui(monkeypatch):
    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    try:
        yield driver
    finally:
        root.destroy()


def _texts(widget):
    out = set()
    for child in widget.winfo_children():
        try:
            out.add(str(child.cget("text")))
        except tk.TclError:
            pass
        out |= _texts(child)
    return out


def test_the_performance_panel_offers_no_gpu_and_no_dead_controls(gui):
    texts = _texts(gui._labelframe(gui.root, "Performance"))
    assert not {"GPU", "GPU chunk:", "Max workers:"} & texts
    assert {"Numba", "Cache baseline", "Seed all", "Seed:", "Unc. workers (0=auto):"} <= texts


def test_a_stored_gpu_choice_never_turns_the_gpu_on(monkeypatch):
    calls = []
    monkeypatch.setattr(performance, "set_gpu", lambda enabled: calls.append(bool(enabled)))
    ui_app.save_config({**ui_app.DEFAULTS, "perf_gpu": True})
    root = tk.Tk()
    try:
        app = ui_app.PeakFitApp(root)
        app.apply_performance()
        root.update()
        assert app._settings().get("perf_gpu", False) is False
    finally:
        root.destroy()
    assert True not in calls
    assert performance.which_backend() != "cupy"


def test_auto_uncertainty_workers_run_sequentially(gui, monkeypatch, tmp_path):
    seen = []

    def spy(**kw):
        seen.append(kw.get("workers"))
        theta = np.asarray(kw["theta"], float)
        stats = {f"p{i}": {"est": float(t), "sd": 0.1, "p2.5": float(t) - 0.2, "p97.5": float(t) + 0.2} for i, t in enumerate(theta)}
        return core_uncertainty.UncertaintyResult("bootstrap", "Bootstrap", stats, {"alpha": 0.05}, None)

    import batch.runner as runner

    for module in (core_uncertainty, runner):
        monkeypatch.setattr(module, "bootstrap_ci", spy, raising=False)
    try:
        import core.pipeline as pipeline
    except ImportError:
        pass
    else:
        monkeypatch.setattr(pipeline, "bootstrap_ci", spy, raising=False)

    path = tmp_path / "spec.csv"
    write_spectrum(path, *mos2_spectrum(2))
    gui.open_file(path)
    gui.set_window(*WINDOW)
    gui.app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]
    gui.app.perf_unc_workers.set(0)
    gui.app.unc_method.set("Bootstrap")
    gui.fit()
    gui.run_uncertainty()
    assert seen and all(w in (None, 1) for w in seen), seen
