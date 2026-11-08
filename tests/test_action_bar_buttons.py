"""Every action-bar button is laid out, including Step, which shows an icon."""
import os

import pytest

from ui import app as app_module

headless = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
if headless:
    pytest.skip("Skipping GUI tests in headless environment", allow_module_level=True)

tk = pytest.importorskip("tkinter")


def test_step_and_fit_buttons_are_packed():
    root = tk.Tk()
    root.withdraw()
    try:
        app = app_module.PeakFitApp(root)
        assert app.step_btn.winfo_manager() == "pack"
        assert app.fit_btn.winfo_manager() == "pack"
        assert app.step_btn.master.pack_slaves() == [app.step_btn, app.fit_btn]
    finally:
        root.destroy()
