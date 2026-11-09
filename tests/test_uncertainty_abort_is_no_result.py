"""An aborted bootstrap or Bayesian run has no result.

The 2026-11-08 review of #253-#255: an aborted run was shown, cached and exported
as complete, from the draws made before the abort (an aborted Bayesian run
exported a header-only CSV). On ea2d8c5 it yielded nothing. Now
core.pipeline.compute_uncertainty raises UncertaintyAborted, so Run shows and
keeps nothing, Export writes the fit and the trace only and says why, and batch
writes no uncertainty for the file it was on.

Run, Export and Run Batch also shared one abort event and one Abort button: a job
that ended disabled the button and cleared the event of another still running.
Each job now has its own event, Abort sets every running job's, and the button
stays enabled while any job runs. Closing the window during an Export killed its
worker thread wherever it was; the window now waits for the aborted job.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from batch import runner
from core import fit_api, pipeline
from core.peaks import Peak
from tests.conftest import bayes_knobs
from tests.gui_driver import Driver
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum
from tests.test_batch_file_isolation import _config, _spectrum, _write


class AbortAfter(threading.Event):
    """An abort event that the user sets after ``n`` looks at it."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def is_set(self) -> bool:
        self.n -= 1
        if self.n < 0:
            self.set()
        return super().is_set()


def _fit(tmp_path, **extra):
    x, y = _spectrum()
    settings = {**_config(tmp_path), "perf_unc_workers": 1, **extra}
    seeds = [Peak(41.0, 90.0, 4.5, 0.5), Peak(59.0, 70.0, 5.5, 0.5)]
    mask = (x >= 20.0) & (x <= 80.0)
    fit = fit_api.run_fit_consistent(
        x, y, seeds, settings, np.full_like(x, 10.0), "add", mask,
        return_jacobian=True, return_predictors=True,
    )
    assert fit["fit_ok"]
    return x, y, mask, fit, settings


def _expect_no_result(method, fit, x, y, mask, settings, event):
    try:
        res = pipeline.compute_uncertainty(fit, x, y, mask, settings, method, abort_event=event)
    except Exception as exc:
        assert type(exc).__name__ == "UncertaintyAborted", repr(exc)
        return
    diag = (res.get("diagnostics") if isinstance(res, dict) else getattr(res, "diagnostics", {})) or {}
    pytest.fail(f"an aborted {method} run returned a result: {dict((k, diag.get(k)) for k in ('aborted', 'n_success', 'n_draws'))}")


def test_an_aborted_bootstrap_returns_no_result(tmp_path):
    x, y, mask, fit, settings = _fit(tmp_path, bootstrap_n=200)
    _expect_no_result("bootstrap", fit, x, y, mask, settings, AbortAfter(6))


def test_an_aborted_bayesian_run_returns_no_result(tmp_path):
    x, y, mask, fit, settings = _fit(tmp_path, **bayes_knobs(walkers=0, burn=200, steps=4000))
    _expect_no_result("bayesian", fit, x, y, mask, settings, AbortAfter(2))


def test_abort_drops_the_draws_not_started(tmp_path, monkeypatch):
    # With a pool of workers the draws already queued used to run to the end,
    # so Abort (or closing the window) waited for all of them.
    x, y, mask, fit, settings = _fit(tmp_path, bootstrap_n=400, perf_unc_workers=4)
    calls = []
    real = fit_api.run_fit_consistent

    def counted(*args, **kwargs):
        calls.append(1)
        time.sleep(0.002)
        return real(*args, **kwargs)

    monkeypatch.setattr(fit_api, "run_fit_consistent", counted)
    with pytest.raises(Exception):
        pipeline.compute_uncertainty(fit, x, y, mask, settings, "bootstrap", abort_event=AbortAfter(8))
    assert len(calls) < 100, f"{len(calls)} of 400 refits ran after the abort"


def test_batch_writes_no_uncertainty_for_the_file_it_was_on(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    for name in ("a.csv", "b.csv"):
        _write(in_dir / name, *_spectrum())
    cfg = {**_config(out_dir), "bootstrap_n": 200, "perf_unc_workers": 1}

    res = runner.run_batch(
        [str(in_dir / "*.csv")], cfg, compute_uncertainty=True, unc_method="Bootstrap",
        abort_event=AbortAfter(6),  # the batch looks once, then the first file's draws
    )

    assert isinstance(res, dict) and res["aborted"]
    rows = {r["file"]: r for r in _rows(out_dir / "batch_files.csv")}
    assert (rows["a.csv"]["status"], rows["a.csv"]["unc_status"]) == ("ok", "aborted")
    assert rows["b.csv"]["status"] == "not_processed"
    assert sorted(p.name for p in out_dir.iterdir() if p.name.startswith("a_")) == ["a_fit.csv"]


def _rows(path):
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- the GUI -------------------------------------------------------------------------

needs_display = pytest.mark.skipif(
    os.environ.get("DISPLAY", "") == "" and os.name != "nt",
    reason="drives the real GUI: needs a display (run under xvfb-run)",
)


@pytest.fixture
def gui(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    from ui import app as ui_app

    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    path = tmp_path / "spec.csv"
    write_spectrum(path, *mos2_spectrum(6))
    driver.open_file(path)
    driver.set_window(*WINDOW)
    app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]
    app.refresh_tree()
    driver.fit()
    try:
        yield driver
    finally:
        for evt in list(getattr(app, "_jobs", ())) + [getattr(app, "_abort_evt", None)]:
            if evt is not None:
                evt.set()
        try:
            root.destroy()
        except tk.TclError:
            pass  # the test closed the window


def _slow_bootstrap(gui, monkeypatch, n=2000):
    """A bootstrap of many slow refits: it runs until the test aborts it."""

    gui.app._unc_boot_radio.invoke()
    gui.app.boot_n_var.set(n)
    real = fit_api.run_fit_consistent

    def slow(*args, **kwargs):
        time.sleep(0.005)
        return real(*args, **kwargs)

    monkeypatch.setattr(fit_api, "run_fit_consistent", slow)


@needs_display
def test_an_aborted_run_shows_and_keeps_nothing(gui, monkeypatch, tmp_path):
    _slow_bootstrap(gui, monkeypatch)
    gui.root.after(500, gui.app.abort_btn.invoke)
    line = gui.run_uncertainty()

    assert "aborted" in line.lower(), line
    assert gui.app.last_uncertainty is None and gui.app._unc_cache is None
    assert str(gui.app.abort_btn.cget("state")) == "disabled"

    # Nothing was kept, so Export computes its own; abort that too.
    out = tmp_path / "out"
    out.mkdir()
    gui.root.after(500, gui.app.abort_btn.invoke)
    gui.export(out / "A.csv")
    assert sorted(p.name for p in out.iterdir()) == ["A_fit.csv", "A_trace.csv"]
    assert any("the run was aborted" in ln for ln in gui.app._log_buffer)


@needs_display
def test_abort_still_stops_a_run_after_an_export_ends(gui, monkeypatch, tmp_path):
    _slow_bootstrap(gui, monkeypatch)
    n0 = len(gui.app._log_buffer)
    gui.click("Run", within="Uncertainty")  # the slow bootstrap starts
    gui.app._unc_asym_radio.invoke()
    out = tmp_path / "out"
    out.mkdir()
    gui.export(out / "E.csv")  # an asymptotic Export, done in a moment

    assert str(gui.app.abort_btn.cget("state")) == "normal", "the Run is still going"
    gui.app.abort_btn.invoke()

    def ended():
        return [ln for ln in gui.app._log_buffer[n0:] if "Uncertainty aborted" in ln or "Computed Bootstrap" in ln]

    gui.pump_until(lambda: bool(ended()), timeout=300, what="the Run")
    assert "aborted" in ended()[0], ended()[0]


@needs_display
def test_closing_the_window_during_an_export_waits_for_its_files(gui, monkeypatch, tmp_path):
    _slow_bootstrap(gui, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    root = gui.root
    events = []
    real_write = pipeline.write_spectrum

    def write(*args, **kwargs):
        events.append("write")
        written = real_write(*args, **kwargs)
        events.append("written")
        return written

    monkeypatch.setattr(pipeline, "write_spectrum", write)
    root.bind("<Destroy>", lambda e: events.append("destroy") if str(e.widget) == "." else None, add="+")

    def close():  # what the window manager's close button runs (Tk's own default: destroy)
        root.tk.call(root.protocol("WM_DELETE_WINDOW"))

    gui.save_answers.append(str(out / "C.csv"))
    gui.click("Export")
    root.after(500, close)
    root.mainloop()  # until the window is gone

    t0 = time.monotonic()
    old_event = getattr(gui.app, "_abort_evt", None)
    if old_event is not None:
        old_event.set()  # let a worker the window left behind finish
    while "written" not in events and time.monotonic() - t0 < 300:
        time.sleep(0.05)
    assert events == ["write", "written", "destroy"], events
    assert sorted(p.name for p in out.iterdir()) == ["C_fit.csv", "C_trace.csv"]
