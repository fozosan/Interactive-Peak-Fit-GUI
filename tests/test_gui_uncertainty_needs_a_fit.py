"""Uncertainty is of a fit: the GUI computes it as batch does, for a converged fit.

The one uncertainty context (game plan 2.1) is built from the fit: its
residuals, Jacobian, locks, bounds and weights. Batch computes it only for a
fit that converged. The GUI used to compute intervals for any peaks on screen,
fitted or not, which are no estimate of anything. Now Run and Export compute
the uncertainty of the current fit; with no current fit (none yet, or the
peaks, data, baseline or window changed since) they say so, and Export still
writes the peak table (fit_ok False) and the trace.
"""

from __future__ import annotations

import os

import pytest

from tests.gui_driver import Driver
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from ui import app as ui_app  # noqa: E402


@pytest.fixture
def gui(monkeypatch, tmp_path):
    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    path = tmp_path / "spec.csv"
    write_spectrum(path, *mos2_spectrum(6))
    driver.open_file(path)
    driver.set_window(*WINDOW)
    app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]
    app.refresh_tree()
    try:
        yield driver
    finally:
        root.destroy()


def test_run_waits_for_a_fit(gui):
    line = gui.run_uncertainty()
    assert "Uncertainty not computed" in line and "Fit" in line
    assert gui.app.last_uncertainty is None

    gui.fit()
    assert "Computed" in gui.run_uncertainty()
    assert gui.app.last_uncertainty is not None

    gui.app.peaks[0].center += 0.5  # moved by hand: no longer the fit
    assert "Uncertainty not computed" in gui.run_uncertainty()


def test_export_without_a_fit_writes_the_table_and_the_trace_only(gui, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gui.export(out / "spec.csv")
    assert sorted(p.name for p in out.iterdir()) == ["spec_fit.csv", "spec_trace.csv"]
    assert any("Uncertainty not computed" in line for line in gui.app._log_buffer)
