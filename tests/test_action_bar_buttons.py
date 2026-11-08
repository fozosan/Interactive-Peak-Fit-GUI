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


def test_step_icon_is_drawn_in_the_button_text_colour():
    """A dark theme has light button text. A black icon on its dark button barely shows."""
    pytest.importorskip("sv_ttk")
    from ui.theme import apply_theme

    root = tk.Tk()
    root.withdraw()
    try:
        apply_theme(root)
        app = app_module.PeakFitApp(root)
        text = str(app._style.lookup("TButton", "foreground"))
        text_rgb = tuple(v // 257 for v in root.winfo_rgb(text))
        icon = app.step_icon
        opaque = {
            icon.get(x, y)
            for y in range(icon.height())
            for x in range(icon.width())
            if not icon.transparency_get(x, y)
        }
        assert opaque == {text_rgb}
    finally:
        root.destroy()
