"""Export and batch never delete a file the app cannot show it wrote.

The 2026-11-08 review of #253-#255: "Export as G" deleted G_uncertainty.csv,
G_uncertainty_wide.csv, G_uncertainty.txt, G_uncertainty_band.csv and G_band.csv
before writing, whether or not it wrote uncertainty, and a batch with uncertainty
off deleted them too. G_band.csv is no output of the GUI: it was the user's file.

Now the files an export writes replace their namesakes, and of the uncertainty
files it does not write, only those whose first line shows they are the app's own
output (an earlier fit's) are deleted, and only when it computes uncertainty.
Anything else is left, and the log says so.
"""

from __future__ import annotations

import os

import pytest

from batch import runner
from core import data_io, pipeline
from tests.gui_driver import Driver, new_root
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum
from tests.test_batch_file_isolation import _config, _spectrum, _write

USER_TEXT = "my own notes, not the app's\n"
UNC = ("_uncertainty.csv", "_uncertainty_wide.csv", "_uncertainty.txt", "_uncertainty_band.csv")


def test_only_the_apps_own_uncertainty_files_count_as_its_own(tmp_path):
    own = {
        "a_uncertainty.csv": "file,peak,param,value,stderr,p24_5,p75_5,ci_lo,ci_hi,method,rmse,dof,flags\n",
        "a_uncertainty_wide.csv": "file,peak,method,rmse,dof,center,center_stderr,center_ci_lo,center_ci_hi,height\n",
        "a_uncertainty.txt": "Uncertainty method: Bootstrap\nPeak 1: ...\n",
        "a_uncertainty_band.csv": "x,y_lo51,y_hi51\n1,2,3\n",
        "b_uncertainty_band.csv": "x,y_fit,y_lo95,y_hi95\n1,2,3,4\n",  # an older band
    }
    foreign = {
        "c_uncertainty.csv": USER_TEXT,
        "c_uncertainty.txt": "Notes on the uncertainty of sample C\n",
        "c_uncertainty_band.csv": "x,band_lo,band_hi\n",
        "c_band.csv": "x,y_lo95,y_hi95\n",  # not an uncertainty file name at all
        "c_fit.csv": "file,peak,param,value,stderr,p2_5,p97_5,ci_lo,ci_hi,method,\n",
    }
    for name, text in {**own, **foreign}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "d_uncertainty.csv").write_bytes(b"\xff\xfe binary")
    assert all(data_io.is_own_uncertainty_file(tmp_path / n) for n in own)
    assert not any(data_io.is_own_uncertainty_file(tmp_path / n) for n in foreign)
    assert not data_io.is_own_uncertainty_file(tmp_path / "d_uncertainty.csv")
    assert not data_io.is_own_uncertainty_file(tmp_path / "missing_uncertainty.csv")


# --- batch -----------------------------------------------------------------------


def _batch(in_dir, out_dir, compute, method="Asymptotic"):
    return runner.run_batch(
        [str(in_dir / "a.csv")], _config(out_dir), compute_uncertainty=compute, unc_method=method,
        log=lambda msg: LOG.append(msg),
    )


LOG: list[str] = []


@pytest.fixture
def folders(tmp_path):
    LOG.clear()
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write(in_dir / "a.csv", *_spectrum())
    return in_dir, out_dir


def test_batch_with_uncertainty_off_deletes_nothing(folders):
    in_dir, out_dir = folders
    _batch(in_dir, out_dir, compute=True)  # an earlier run: the app's own files
    own = {s: (out_dir / f"a{s}").read_text(encoding="utf-8") for s in UNC if (out_dir / f"a{s}").exists()}
    assert set(own) == {"_uncertainty.csv", "_uncertainty.txt", "_uncertainty_band.csv"}
    (out_dir / "a_uncertainty_wide.csv").write_text(USER_TEXT, encoding="utf-8")
    (out_dir / "a_band.csv").write_text(USER_TEXT, encoding="utf-8")

    ok, n = _batch(in_dir, out_dir, compute=False)

    assert (ok, n) == (1, 1)
    for suffix, text in own.items():
        assert (out_dir / f"a{suffix}").read_text(encoding="utf-8") == text, suffix
    assert (out_dir / "a_uncertainty_wide.csv").read_text(encoding="utf-8") == USER_TEXT
    assert (out_dir / "a_band.csv").read_text(encoding="utf-8") == USER_TEXT
    # and the log says which files no longer describe the fit beside them
    left = [m for m in LOG if m.startswith("left a_uncertainty")]
    assert len(left) == 4, LOG


def test_batch_rerun_replaces_its_own_files_and_keeps_the_users(folders):
    in_dir, out_dir = folders
    _batch(in_dir, out_dir, compute=True, method="Bootstrap")  # writes a band too
    assert (out_dir / "a_uncertainty_band.csv").exists()
    (out_dir / "a_band.csv").write_text(USER_TEXT, encoding="utf-8")
    (out_dir / "a_uncertainty_wide.csv").write_text(USER_TEXT, encoding="utf-8")

    cfg = _config(out_dir)
    cfg["ui_band_pref_bootstrap"] = False  # this run writes no band
    LOG.clear()
    runner.run_batch([str(in_dir / "a.csv")], cfg, compute_uncertainty=True, unc_method="Bootstrap",
                     log=lambda m: LOG.append(m))

    assert (out_dir / "a_uncertainty.csv").read_text(encoding="utf-8").startswith("file,peak,param,value,")
    assert not (out_dir / "a_uncertainty_band.csv").exists()  # the earlier run's band: the app's own
    assert any(m.startswith("removed a_uncertainty_band.csv") for m in LOG), LOG
    assert (out_dir / "a_band.csv").read_text(encoding="utf-8") == USER_TEXT
    assert (out_dir / "a_uncertainty_wide.csv").read_text(encoding="utf-8") == USER_TEXT
    assert any(m.startswith("left a_uncertainty_wide.csv") for m in LOG), LOG


# --- the GUI's Export ----------------------------------------------------------------

gui_only = pytest.mark.skipif(
    os.environ.get("DISPLAY", "") == "" and os.name != "nt",
    reason="drives the real GUI: needs a display (run under xvfb-run)",
)


@pytest.fixture
def gui(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    from ui import app as ui_app

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


@gui_only
def test_export_keeps_files_it_did_not_write(gui, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "G_band.csv").write_text(USER_TEXT, encoding="utf-8")
    (out / "G_uncertainty.csv").write_text(USER_TEXT, encoding="utf-8")

    gui.export(out / "G.csv")  # no fit yet: no uncertainty

    assert (out / "G_band.csv").read_text(encoding="utf-8") == USER_TEXT
    assert (out / "G_uncertainty.csv").read_text(encoding="utf-8") == USER_TEXT
    assert any("left G_uncertainty.csv" in line for line in gui.app._log_buffer)


@gui_only
def test_export_retires_only_its_own_stale_uncertainty(gui, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gui.fit()
    gui.app.cfg["export_unc_wide"] = True
    gui.export(out / "G.csv")  # the asymptotic set, wide and band included
    assert all((out / f"G{suffix}").exists() for suffix in UNC)
    (out / "G_band.csv").write_text(USER_TEXT, encoding="utf-8")

    gui.app.peaks[0].center += 0.5  # no longer the fit: this export writes no uncertainty
    gui.export(out / "G.csv")

    assert not any((out / f"G{suffix}").exists() for suffix in UNC)
    assert (out / "G_band.csv").read_text(encoding="utf-8") == USER_TEXT
    assert sorted(p.name for p in out.iterdir()) == ["G_band.csv", "G_fit.csv", "G_trace.csv"]
