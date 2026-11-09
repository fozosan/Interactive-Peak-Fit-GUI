"""The settings file lives beside the code, in ui/ui_config/, so OneDrive syncs it.

OneDrive syncs the repo folder between the Linux and Windows machines; $HOME is
neither synced nor backed up. Settings used to live in ~/.gl_peakfit_config.json.
While the new file does not exist, that old file is read instead, and nothing
ever writes to it. GL_PEAKFIT_CONFIG names another file, for tests and tools.
ui/app.py and the v2.7 gl_decon_gui_batch.py share the one file.
"""
import importlib
import json
import os
from pathlib import Path

import pytest

from ui import app

REPO = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO / "ui" / "ui_config" / "gl_peakfit_config.json"
HEADLESS = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
OLD_SETTINGS = {
    "x_label": "Raman shift",
    "templates": {"MoS2": [{"center": 383.0, "height": 1.0, "fwhm": 2.5, "eta": 0.5}]},
}


@pytest.fixture(params=["ui.app", "gl_decon_gui_batch"])
def config_module(request):
    """Each module that reads and writes the settings file."""
    if request.param == "gl_decon_gui_batch" and HEADLESS:
        pytest.skip("gl_decon_gui_batch selects the TkAgg backend when imported")
    return importlib.import_module(request.param)


@pytest.fixture
def old_file(tmp_path):
    """An old ~/.gl_peakfit_config.json, and a stamp to prove it was never written."""
    path = tmp_path / "home" / ".gl_peakfit_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps(OLD_SETTINGS))
    return path, (path.read_bytes(), path.stat().st_mtime_ns)


def test_the_default_file_is_under_ui_config(config_module, monkeypatch):
    monkeypatch.delenv("GL_PEAKFIT_CONFIG", raising=False)
    assert config_module._config_paths() == (REPO_CONFIG, Path.home() / ".gl_peakfit_config.json")


def test_gl_peakfit_config_wins(config_module, monkeypatch, tmp_path):
    chosen = tmp_path / "chosen.json"
    monkeypatch.setenv("GL_PEAKFIT_CONFIG", str(chosen))
    # An explicit file is used as given, with no fallback to the old home file.
    assert config_module._config_paths() == (chosen, None)


def test_the_old_file_is_read_until_the_new_one_exists_and_never_written(
    config_module, monkeypatch, tmp_path, old_file
):
    old, stamp = old_file
    new = tmp_path / "ui_config" / "gl_peakfit_config.json"
    monkeypatch.setattr(config_module, "CONFIG_PATH", new)
    monkeypatch.setattr(config_module, "LEGACY_CONFIG_PATH", old)

    cfg = config_module.load_config()
    assert cfg["x_label"] == "Raman shift"
    assert cfg["templates"] == OLD_SETTINGS["templates"]
    assert not new.exists()

    cfg["x_label"] = "changed"
    config_module.save_config(cfg)
    assert json.loads(new.read_text())["x_label"] == "changed"
    assert config_module.load_config()["x_label"] == "changed"
    assert (old.read_bytes(), old.stat().st_mtime_ns) == stamp


def test_the_first_save_creates_the_folder(config_module, monkeypatch, tmp_path):
    new = tmp_path / "ui_config" / "gl_peakfit_config.json"
    monkeypatch.setattr(config_module, "CONFIG_PATH", new)
    config_module.save_config({**config_module.DEFAULTS, "x_label": "saved"})
    assert json.loads(new.read_text())["x_label"] == "saved"


@pytest.mark.skipif(HEADLESS, reason="requires display")
def test_the_first_start_carries_the_old_settings_into_the_new_file(monkeypatch, tmp_path, old_file):
    tk = pytest.importorskip("tkinter")
    old, stamp = old_file
    new = tmp_path / "ui_config" / "gl_peakfit_config.json"
    monkeypatch.setattr(app, "CONFIG_PATH", new)
    monkeypatch.setattr(app, "LEGACY_CONFIG_PATH", old)

    root = tk.Tk()
    root.withdraw()
    try:
        app.PeakFitApp(root)
    finally:
        root.destroy()

    saved = json.loads(new.read_text())
    assert saved["x_label"] == "Raman shift"
    assert list(saved["templates"]) == ["MoS2"]
    assert (old.read_bytes(), old.stat().st_mtime_ns) == stamp


def test_the_suite_never_points_at_the_repo_file():
    # conftest sets GL_PEAKFIT_CONFIG before any test imports the app, so every
    # import, a fresh re-import included, resolves to a temporary file.
    path, old = app._config_paths()
    assert not path.resolve().is_relative_to(REPO)
    assert old is None
    assert not app.CONFIG_PATH.resolve().is_relative_to(REPO)
