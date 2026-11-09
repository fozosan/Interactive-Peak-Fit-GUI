"""One settings object: what the GUI fits and exports with is what Run Batch sends.

The 2026-11-07 review found that batch ignored the GUI's Use baseline (areas off
by 119 to 172x), its alpha and its wide-CSV choice, and that the Modern solvers'
restarts never reached batch: the button built its own partial config.
"""

from __future__ import annotations

import os

import pytest

from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from tests.gui_driver import Driver  # noqa: E402
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


def test_run_batch_sends_the_settings_the_gui_fits_and_exports_with(gui, monkeypatch, tmp_path):
    path = tmp_path / "data" / "spec.csv"
    path.parent.mkdir()
    write_spectrum(path, *mos2_spectrum(5))
    gui.open_file(path)
    gui.set_window(*WINDOW)
    gui.app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]

    gui.app.alpha_var.set(0.1)
    gui.app.cfg["export_unc_wide"] = True
    gui.app.use_baseline.set(False)
    gui.app.solver_choice.set("modern_trf")
    gui.app.modern_restarts.set(3)
    gui.app.modern_jitter.set(5.0)
    gui.app.batch_source.set("current")

    seen = []
    monkeypatch.setattr(
        ui_app.batch_runner, "run_batch", lambda patterns, config, **kw: (seen.append(dict(config)), (1, 1))[1]
    )
    gui.run_batch(path.parent, tmp_path / "batch_summary.csv")

    [config] = seen
    assert config["unc_alpha"] == pytest.approx(0.1)
    assert config["export_unc_wide"] is True
    assert config["use_baseline"] is False
    assert config["solver"] == "modern_trf"
    assert config["solver_restarts"] == 3
    assert config["solver_jitter_pct"] == pytest.approx(5.0)
    assert (config["fit_xmin"], config["fit_xmax"]) == WINDOW
