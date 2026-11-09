"""The GUI's uncertainty defaults are the core's and batch's.

- Bootstrap start jitter: 0 (PR #248). A start jitter scales with each
  parameter's value, so 2 % moves a centre at 383 cm-1 by 7.7 cm-1, the refits
  land in other minima and the intervals came out 4 to 4500x too wide. A file
  from before this version holds the old default, which the app wrote into
  every file at startup, or a value chosen on that scale before this was
  measured (the user's holds 3 %). It starts at 0, once, and the log says so;
  a jitter set after that is the user's and is kept.
- Bayesian steps: 8000 (PR #252: at 4000, 29 to 43 % of two-peak chains were
  shorter than 50 autocorrelation times). The old default, 4000, was also
  written into every file, and gives way the same way; another stored value is
  kept.
- Bootstrap residuals are studentized, as batch does since PR #252.
"""

from __future__ import annotations

import json
import os

import pytest

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("starts the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from ui import app as ui_app  # noqa: E402


OLD = {k: v for k, v in ui_app.DEFAULTS.items() if k != "settings_version"}  # a file from before


def _start():
    root = tk.Tk()
    return root, ui_app.PeakFitApp(root)


def _saved():
    return json.loads(ui_app.CONFIG_PATH.read_text())


def test_a_fresh_start_uses_the_core_defaults():
    root, app = _start()
    try:
        settings = app._settings()
        assert app.bootstrap_jitter_var.get() == 0.0 and settings["bootstrap_jitter"] == 0.0
        assert app.bayes_steps_var.get() == 8000 and settings["bayes_steps"] == 8000
        assert app.boot_studentize_var.get() is True and settings["boot_studentize"] is True
    finally:
        root.destroy()


@pytest.mark.parametrize("stored_steps, steps_now", [(4000, 8000), (6000, 6000)])
def test_an_old_file_s_jitter_and_old_default_steps_give_way_once(stored_steps, steps_now):
    ui_app.save_config({**OLD, "bootstrap_jitter": 0.03, "bayes_steps": stored_steps, "bayes_burn": 1500})
    root, app = _start()
    try:
        assert app.bootstrap_jitter_var.get() == 0.0
        assert [line for line in app._log_buffer if "bootstrap start jitter is now 0" in line and "(it was 3 %)" in line]
        assert app._settings()["bootstrap_jitter"] == 0.0
        assert app.bayes_steps_var.get() == steps_now  # only the old default gives way
        assert app.bayes_burn_var.get() == 1500  # not a changed default: kept
        # the user sets a jitter and other steps in this version
        app.bootstrap_jitter_var.set(1.0)
        app._safe_jitter_pct()
        app.bayes_steps_var.set(4000)
        app._settings()
    finally:
        root.destroy()
    saved = _saved()
    assert saved["bootstrap_jitter"] == pytest.approx(0.01)

    root, app = _start()
    try:
        assert not [line for line in app._log_buffer if "bootstrap start jitter" in line]  # said once
        assert app.bootstrap_jitter_var.get() == pytest.approx(1.0)  # kept now
        assert app.bayes_steps_var.get() == 4000  # chosen now, so kept
    finally:
        root.destroy()


def test_a_stored_studentize_choice_is_kept():
    ui_app.save_config({**ui_app.DEFAULTS, "boot_studentize": False})
    root, app = _start()
    try:
        assert app.boot_studentize_var.get() is False
        assert app._settings()["boot_studentize"] is False
    finally:
        root.destroy()
