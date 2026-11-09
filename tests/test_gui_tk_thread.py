"""Every Tk call runs on the main thread, and Export never computes there.

Game plan 3.4: 36 to 73 Tk calls per uncertainty run came from worker threads,
and one fit result was lost. Tk serves another thread only while the main loop
runs, so such calls fail outright otherwise ("main thread is not in main
loop"). The app's workers now hand everything to the Tk thread, which polls for
them with root.after. Game plan 2.1: Export computed its uncertainty on the Tk
thread and froze the window for 7 minutes.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from tests.gui_driver import Driver
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from core import uncertainty as core_uncertainty  # noqa: E402
from ui import app as ui_app  # noqa: E402

TK_DIR = os.path.dirname(tk.__file__)


@pytest.fixture
def gui(monkeypatch):
    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    try:
        yield driver
    finally:
        root.destroy()


@pytest.fixture
def spectrum(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "spec.csv"
    write_spectrum(path, *mos2_spectrum(9))
    return path


def _open_and_seed(gui, path):
    gui.open_file(path)
    gui.set_window(*WINDOW)
    gui.app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]
    gui.app.refresh_tree()


def _patch_bootstrap(monkeypatch, fake):
    import batch.runner as runner

    for module in (core_uncertainty, runner):
        monkeypatch.setattr(module, "bootstrap_ci", fake, raising=False)
    try:
        import core.pipeline as pipeline
    except ImportError:
        return
    monkeypatch.setattr(pipeline, "bootstrap_ci", fake, raising=False)


def _canned(*, theta, x_all, alpha=0.05, **_):
    theta = np.asarray(theta, float)
    stats = {f"p{i}": {"est": float(t), "sd": 0.1, "p2.5": float(t) - 0.2, "p97.5": float(t) + 0.2} for i, t in enumerate(theta)}
    x = np.asarray(x_all, float)
    return core_uncertainty.UncertaintyResult(
        method="bootstrap", label="Bootstrap", stats=stats, diagnostics={"alpha": float(alpha)},
        band=(x, x * 0.0 + 500.0, x * 0.0 + 520.0),
    )


def test_no_tk_call_leaves_the_main_thread(gui, tmp_path, spectrum):
    """Fit, Uncertainty, Export and Run Batch, with every worker thread watched."""

    off_main = []

    def watch(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(TK_DIR):
            off_main.append((threading.current_thread().name, frame.f_code.co_name))
        elif event == "c_call" and type(getattr(arg, "__self__", None)).__name__ == "tkapp":
            off_main.append((threading.current_thread().name, getattr(arg, "__name__", "?")))

    app = gui.app
    _open_and_seed(gui, spectrum)
    app.unc_method.set("Bootstrap")
    app.boot_n_var.set(8)
    app.batch_source.set("current")
    app.batch_unc_enabled.set(True)
    threading.setprofile(watch)  # every thread started from here on
    try:
        gui.fit()
        gui.run_uncertainty()
        gui.export(tmp_path / "spec.csv")
        out = tmp_path / "batch"
        out.mkdir()
        gui.run_batch(spectrum.parent, out / "batch_summary.csv")
    finally:
        threading.setprofile(None)
    assert (tmp_path / "spec_uncertainty.csv").exists() and (tmp_path / "batch" / "spec_fit.csv").exists()
    assert off_main == []


def test_export_computes_off_the_tk_thread_and_the_window_stays_live(gui, monkeypatch, tmp_path, spectrum):
    ran = []

    def slow_bootstrap(**kw):
        ran.append((threading.current_thread() is threading.main_thread(), time.monotonic()))
        time.sleep(1.0)
        ran.append((None, time.monotonic()))
        return _canned(**kw)

    _patch_bootstrap(monkeypatch, slow_bootstrap)
    ticks = []

    def tick():
        ticks.append(time.monotonic())
        gui.root.after(20, tick)

    _open_and_seed(gui, spectrum)
    gui.app.unc_method.set("Bootstrap")
    gui.fit()
    gui.root.after(20, tick)
    gui.export(tmp_path / "spec.csv")

    assert [on_main for on_main, _ in ran if on_main is not None] == [False]
    t_start, t_end = ran[0][1], ran[-1][1]
    assert sum(t_start < t < t_end for t in ticks) > 10  # the main loop ran on
    assert (tmp_path / "spec_uncertainty.csv").exists()
