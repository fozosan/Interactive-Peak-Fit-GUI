"""Regressions 4-7 and 9 of the 2026-11-08 review of #253-#255, in the real GUI.

4. "Fit first" fired after a fresh fit when a table row was selected and the fit
   had left the peaks out of centre order: Run re-sorted them, so the peaks on
   screen were no longer the fit's.
5. Unchecking Compute band no longer suppressed the asymptotic band.
6. The bootstrap and Bayesian progress lines no longer reached the log and the
   status bar (no progress callback was passed).
7. An empty BLAS threads field made Fit, Export, Run and Run Batch do nothing,
   without a word.
9. A typed export name "name.txt" gave name.txt_fit.csv.
"""

from __future__ import annotations

import csv
import os
import threading

import pytest

from tests.gui_driver import Driver, new_root
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from ui import app as ui_app  # noqa: E402


@pytest.fixture
def gui(monkeypatch, tmp_path):
    root = new_root(monkeypatch)
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


def _choose_method(gui, radio):
    radio.invoke()
    assert gui.app._unc_selected_method_key() in str(radio.cget("value")).lower()


def test_a_fresh_fit_stays_current_with_a_row_selected(gui, tmp_path):
    # Seeds out of centre order: the fit keeps their order (as in the review,
    # where a capped peak swapped places), and a row is selected.
    gui.app.peaks = [ui_app.Peak(**dict(p)) for p in reversed(TEMPLATE)]
    gui.app.refresh_tree()
    gui.fit()
    centers = [p.center for p in gui.app.peaks]
    assert centers != sorted(centers)
    gui.app.tree.selection_set(gui.app.tree.get_children()[0])
    gui.root.update()

    assert "Computed" in gui.run_uncertainty()
    assert [p.center for p in gui.app.peaks] == centers
    gui.export(tmp_path / "R.csv")
    with open(tmp_path / "R_fit.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["fit_ok"] for r in rows} == {"True"}


def test_an_edit_in_the_fields_still_counts(gui):
    gui.fit()
    gui.app.tree.selection_set(gui.app.tree.get_children()[0])
    gui.root.update()
    gui.app.center_var.set(gui.app.peaks[0].center + 0.5)  # typed, not applied
    assert "Uncertainty not computed" in gui.run_uncertainty()


def test_compute_band_off_leaves_the_asymptotic_without_a_band(gui, tmp_path):
    gui.fit()
    gui.app.compute_ci_band_var.set(False)
    gui.app._toggle_ci_compute()
    assert "Computed" in gui.run_uncertainty()
    assert gui.app.ci_band is None
    gui.export(tmp_path / "NB.csv")
    assert not (tmp_path / "NB_uncertainty_band.csv").exists()
    assert (tmp_path / "NB_uncertainty.csv").exists()


@pytest.mark.parametrize("method", ["bootstrap", "bayesian"])
def test_progress_lines_reach_the_log_and_the_status_bar(gui, method):
    gui.fit()
    app = gui.app
    if method == "bootstrap":
        _choose_method(gui, app._unc_boot_radio)
        app.boot_n_var.set(40)
        expected = "Bootstrap: "
    else:
        _choose_method(gui, app._unc_bayes_radio)
        app.bayes_walkers_var.set(16)
        app.bayes_burn_var.set(100)
        app.bayes_steps_var.set(400)
        expected = "Bayesian MCMC: "
    shown = []
    app.status_var.trace_add("write", lambda *_: shown.append((threading.current_thread(), app.status_var.get())))
    n0 = len(app._log_buffer)
    assert "Computed" in gui.run_uncertainty()

    lines = [ln for ln in app._log_buffer[n0:] if expected in ln]
    assert lines, app._log_buffer[n0:][:8]
    on_bar = [text for thread, text in shown if expected in text]
    assert on_bar, "no progress line reached the status bar"
    assert {thread for thread, _ in shown} == {app._tk_thread}  # Tk only from its own thread


def test_an_empty_blas_threads_field_falls_back_to_the_default(gui):
    entry_var = gui.app.perf_blas_threads
    entry_var._tk.globalsetvar(entry_var._name, "")  # the user cleared the field
    gui.app.fit()  # raised TclError here before; from the button, nothing happened
    gui.pump_until(lambda: any("Fit finished" in ln for ln in gui.app._log_buffer), what="Fit")
    assert any("BLAS threads ''" in ln and "default" in ln for ln in gui.app._log_buffer)
    assert gui.app.perf_blas_threads.get() == 0


@pytest.mark.parametrize(
    "typed, base",
    [("name.txt", "name"), ("Name.CSV", "Name"), ("x.dat", "x"), ("MoS2_0.5mW", "MoS2_0.5mW"),
     ("MoS2_0.20sec.csv", "MoS2_0.20sec")],
)
def test_a_typed_name_drops_only_a_data_extension(gui, tmp_path, typed, base):
    gui.export(tmp_path / typed)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(("_fit.csv", "_trace.csv"))) == [
        f"{base}_fit.csv", f"{base}_trace.csv"
    ]
