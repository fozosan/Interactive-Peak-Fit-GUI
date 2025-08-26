import os
import tkinter as tk
import numpy as np
import pytest

from ui.app import PeakFitApp, Peak

def has_display():
    try:
        r = tk.Tk()
        r.destroy()
        return True
    except tk.TclError:
        return False

requires_display = pytest.mark.skipif(not has_display(), reason="no display")

def make_app():
    root = tk.Tk()
    root.withdraw()
    app = PeakFitApp(root)
    return root, app

@requires_display
def test_log_buffer_and_console():
    root, app = make_app()
    app.log("hello")
    app.log("world", level="WARN")
    assert len(app._log_buffer) == 2
    assert "hello" in app._log_buffer[0]
    assert "WARN" in app._log_buffer[1]
    assert app._log_console is None
    app.toggle_log()  # open
    text = app._log_console.get("1.0", "end").strip().splitlines()
    assert len(text) == 2
    assert "world" in text[1]
    root.destroy()

@requires_display
def test_log_threadsafe():
    root, app = make_app()
    app.log_threadsafe("bg message")
    root.update()  # process after callbacks
    assert any("bg message" in ln for ln in app._log_buffer)
    root.destroy()

@requires_display
def test_format_asymptotic_summary():
    root, app = make_app()
    app.peaks = [Peak(1.0, 2.0, 3.0, 0.5)]
    cov = np.diag([0.04, 0.25, 0.01, 0.0])
    theta = np.array([1.0, 2.0, 3.0, 0.5])
    rmse = 0.1
    band = (np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    info = {"m": 12, "n": 4, "rank": 4, "dof": 8, "cond": 1.0, "rmse": rmse, "bw": (0.1, 0.2, 0.3), "warn_nonfinite": False}
    lines, warns = app._format_asymptotic_summary(cov, theta, info, band)
    assert any("RMSE" in ln and "dof" in ln and "rank" in ln for ln in lines)
    assert any("Band width" in ln for ln in lines)
    assert any("Peak 1" in ln and "±" in ln for ln in lines)
    assert isinstance(warns, list)
    root.destroy()
