"""A save never leaves a broken settings file, and nothing writes over a file that cannot be read.

The app saves its settings on almost every change and once at every start. save_config used
to write the file in place, and load_config returned the defaults when the file did not parse.
So a save that died part-way left a broken file, and the next start replaced it with the
defaults: the user's peak templates were gone. Now a save writes a temporary file in the same
folder, flushes it to disk and os.replace()s it onto the settings file. A settings file that
cannot be read is renamed aside, to a name that still ends in .json so that the
ui/ui_config/*.json git-ignore rule still covers it, and the user is warned once.
ui/app.py and the v2.7 gl_decon_gui_batch.py share the file, so both are checked.
"""
import importlib
import json
import logging
import os
import sys

import pytest
from tkinter import messagebox

from ui import app

HEADLESS = os.environ.get("DISPLAY", "") == "" and os.name != "nt"
TEMPLATES = {
    f"MoS2-{k:02d}": [
        {"center": 383.0 + k, "height": 10.0 + k, "fwhm": 2.5, "eta": 0.5, "lock_center": False, "lock_width": False}
        for _ in range(4)
    ]
    for k in range(23)
}


@pytest.fixture(params=["ui.app", "gl_decon_gui_batch"])
def config_module(request):
    """Each module that reads and writes the settings file."""
    if request.param == "gl_decon_gui_batch" and HEADLESS:
        pytest.skip("gl_decon_gui_batch selects the TkAgg backend when imported")
    return importlib.import_module(request.param)


@pytest.fixture
def settings(config_module, monkeypatch, tmp_path):
    """The settings file in a folder of its own, so a test can see every file the save leaves."""
    path = tmp_path / "ui_config" / "gl_peakfit_config.json"
    path.parent.mkdir()
    monkeypatch.setattr(config_module, "CONFIG_PATH", path)
    monkeypatch.setattr(config_module, "LEGACY_CONFIG_PATH", None)
    return path


@pytest.fixture
def warnings_shown(monkeypatch):
    shown = []
    monkeypatch.setattr(messagebox, "showwarning", lambda title, msg="", **k: shown.append((title, str(msg))))
    return shown


def _good_settings(config_module):
    return {**config_module.DEFAULTS, "x_label": "Raman shift", "templates": TEMPLATES}


def _aside_files(path):
    return sorted(path.parent.glob(f"{path.stem}.unreadable-*{path.suffix}"))


@pytest.mark.parametrize("dies_in", ["fsync", "replace"])
def test_a_save_that_dies_part_way_leaves_the_previous_file(config_module, settings, warnings_shown, monkeypatch, dies_in):
    config_module.save_config(_good_settings(config_module))
    before = settings.read_bytes()

    def dies(*args, **kwargs):
        raise OSError(f"the save died in os.{dies_in}")

    with monkeypatch.context() as m:
        m.setattr(os, dies_in, dies)
        config_module.save_config({**config_module.DEFAULTS, "x_label": "half-written"})

    assert settings.read_bytes() == before
    assert sorted(p.name for p in settings.parent.iterdir()) == [settings.name]  # no temporary file left
    assert config_module.load_config()["templates"] == TEMPLATES
    assert [t for t, _ in warnings_shown] == ["Config"]


def test_a_save_writes_a_temporary_file_in_the_same_folder_then_replaces(config_module, settings, warnings_shown, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    with monkeypatch.context() as m:
        m.setattr(os, "fsync", lambda fd: (calls.append(("fsync",)), real_fsync(fd))[1])
        m.setattr(os, "replace", lambda src, dst: (calls.append(("replace", src, dst)), real_replace(src, dst))[1])
        config_module.save_config(_good_settings(config_module))

    assert [c[0] for c in calls] == ["fsync", "replace"]
    _, src, dst = calls[1]
    assert os.path.dirname(os.fspath(src)) == os.fspath(settings.parent)
    assert os.fspath(dst) == os.fspath(settings)
    assert sorted(p.name for p in settings.parent.iterdir()) == [settings.name]
    loaded = config_module.load_config()
    assert loaded["templates"] == TEMPLATES and loaded["x_label"] == "Raman shift"
    assert warnings_shown == []


def test_a_replace_refused_for_a_moment_still_saves(config_module, settings, warnings_shown, monkeypatch):
    # Windows refuses to replace a file that another program (a sync client, a virus scanner)
    # holds open for a moment.
    attempts = []
    real_replace = os.replace

    def busy_once(src, dst):
        attempts.append(dst)
        if len(attempts) == 1:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", busy_once)
        config_module.save_config(_good_settings(config_module))

    assert len(attempts) == 2
    assert config_module.load_config()["templates"] == TEMPLATES
    assert warnings_shown == []


def test_an_unreadable_file_is_kept_and_the_start_uses_the_defaults(config_module, settings, warnings_shown, caplog):
    whole = json.dumps(_good_settings(config_module), indent=2)
    settings.write_text(whole[: len(whole) // 2])  # what a save that died part-way used to leave
    broken = settings.read_bytes()

    with caplog.at_level(logging.WARNING):
        cfg = config_module.load_config()

    assert cfg["templates"] == {} and cfg["x_label"] == config_module.DEFAULTS["x_label"]
    [aside] = _aside_files(settings)
    assert aside.read_bytes() == broken
    assert aside.suffix == ".json" and aside.parent == settings.parent  # still git-ignored
    assert not settings.exists()
    assert len(warnings_shown) == 1 and aside.name in warnings_shown[0][1]
    assert sum(r.levelno == logging.WARNING and aside.name in r.getMessage() for r in caplog.records) == 1

    # The start then saves, as the app does; the broken file stays as it was.
    config_module.save_config(cfg)
    assert json.loads(settings.read_text())["templates"] == {}
    assert _aside_files(settings) == [aside] and aside.read_bytes() == broken
    config_module.load_config()
    assert len(warnings_shown) == 1


def test_a_save_never_writes_over_an_unreadable_file(config_module, settings, warnings_shown):
    settings.write_text('{"templates": {"MoS2": [{"center": 383.0')
    broken = settings.read_bytes()

    config_module.save_config(_good_settings(config_module))

    [aside] = _aside_files(settings)
    assert aside.read_bytes() == broken
    assert json.loads(settings.read_text())["templates"] == TEMPLATES
    assert len(warnings_shown) == 1 and aside.name in warnings_shown[0][1]


def test_two_unreadable_files_in_one_second_are_both_kept(config_module, settings, warnings_shown):
    for text in ("{broken one", "{broken two"):
        settings.write_text(text)
        config_module.load_config()
    kept = sorted(p.read_text() for p in _aside_files(settings))
    assert kept == ["{broken one", "{broken two"]


@pytest.mark.skipif(HEADLESS, reason="requires display")
def test_the_app_start_keeps_an_unreadable_file_and_warns_once(monkeypatch, tmp_path, warnings_shown):
    tk = pytest.importorskip("tkinter")
    path = tmp_path / "ui_config" / "gl_peakfit_config.json"
    path.parent.mkdir()
    path.write_text('{"x_label": "Raman shift", "templates": {"MoS2": [')
    broken = path.read_bytes()
    old = tmp_path / "home" / ".gl_peakfit_config.json"  # the old file, read only while the new one is absent
    old.parent.mkdir()
    old.write_text(json.dumps({"x_label": "from the old file"}))
    monkeypatch.setattr(app, "CONFIG_PATH", path)
    monkeypatch.setattr(app, "LEGACY_CONFIG_PATH", old)

    started = []

    class Started(app.PeakFitApp):
        def __init__(self, root, *args, **kwargs):
            super().__init__(root, *args, **kwargs)
            started.append(self)

    monkeypatch.setattr(app, "PeakFitApp", Started)
    monkeypatch.setattr(tk.Tk, "mainloop", lambda self, n=0: None)
    monkeypatch.setattr(sys, "argv", ["run_app.py"])
    try:
        app.main()
    finally:
        for a in started:
            a.root.destroy()

    [aside] = _aside_files(path)
    assert aside.read_bytes() == broken
    saved = json.loads(path.read_text())  # the start saves at once
    assert saved["x_label"] == app.DEFAULTS["x_label"]  # the defaults, not the old file
    assert started[0].cfg["x_label"] == app.DEFAULTS["x_label"]
    assert len(warnings_shown) == 1
