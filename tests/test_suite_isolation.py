"""The suite never touches the user's real settings and never waits on a Tk dialog.

``ui.app`` builds ``CONFIG_PATH`` when it is imported, so conftest points
``GL_PEAKFIT_CONFIG`` into a temporary HOME before any test module imports the
app (test_config_location.py covers the real location). It also answers Tk
dialogs without opening them, and it resets both before every test. The paired tests below run in file order: the first leaves state
behind and the second checks that it is gone.
"""
import os
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from ui import app


def _account_home():
    """Home directory from the account database, which no environment variable moves."""
    try:
        import pwd
    except ImportError:  # Windows has no pwd module
        return None
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def test_config_path_is_in_a_temporary_home():
    temp_root = Path(tempfile.gettempdir()).resolve()
    assert Path.home().resolve().is_relative_to(temp_root)
    assert app.CONFIG_PATH.resolve().is_relative_to(temp_root)
    real_home = _account_home()
    if real_home is not None:
        assert app.CONFIG_PATH.resolve() != (real_home / ".gl_peakfit_config.json").resolve()


def test_a_test_saves_a_setting():
    app.save_config({**app.DEFAULTS, "solver_choice": "classic"})
    assert app.CONFIG_PATH.exists()


def test_the_next_test_starts_without_it():
    assert not app.CONFIG_PATH.exists()


def test_a_test_replaces_a_dialog_without_monkeypatch():
    messagebox.showinfo = lambda *args, **kwargs: "replaced"


def test_the_next_test_gets_the_stock_dialog_answers():
    assert messagebox.showinfo("title", "message") == "ok"
    assert messagebox.showerror("title", "message") == "ok"
    assert messagebox.askyesno("title", "message") is True
    assert simpledialog.askstring("title", "prompt") is None
    assert filedialog.asksaveasfilename() == ""
