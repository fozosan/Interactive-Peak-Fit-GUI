"""main() applies the theme and the font policy before it builds the app."""
import os
import sys
import tkinter.font as tkfont

import pytest

import ui.font_diagnostics as font_diagnostics
from ui import app as app_module

headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


@pytest.fixture
def run_main(monkeypatch):
    """Run main() up to the point where it would enter the Tk main loop."""
    roots = []

    class _App:  # stands in for PeakFitApp, which main() builds after the font policy
        def __init__(self, root):
            roots.append(root)

    monkeypatch.setattr(app_module, "PeakFitApp", _App)
    monkeypatch.setattr(tk.Tk, "mainloop", lambda self, n=0: None)
    monkeypatch.setattr(sys, "argv", ["run_app.py"])
    try:
        yield lambda: (app_module.main(), roots[-1])[1]
    finally:
        for root in roots:
            root.destroy()


def test_main_sets_the_configured_font_size(run_main):
    app_module.save_config({**app_module.DEFAULTS, "ui_font_size": 13})
    root = run_main()
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
        assert tkfont.nametofont(name, root=root).cget("size") == 13


def test_font_diagnostics_key_runs_the_diagnostics(run_main, monkeypatch):
    calls = []
    monkeypatch.setattr(font_diagnostics, "run_diagnostics", lambda root, log_fn=print: calls.append("run") or {})
    monkeypatch.setattr(font_diagnostics, "show_visual_sample", lambda root: calls.append("sample"))

    app_module.save_config({**app_module.DEFAULTS, "font_diagnostics": True})
    run_main()
    assert calls == ["run", "sample"]


def test_no_diagnostics_by_default(run_main, monkeypatch):
    calls = []
    monkeypatch.setattr(font_diagnostics, "run_diagnostics", lambda root, log_fn=print: calls.append("run") or {})
    monkeypatch.setattr(font_diagnostics, "show_visual_sample", lambda root: calls.append("sample"))

    run_main()
    assert calls == []
