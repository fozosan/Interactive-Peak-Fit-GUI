"""The exported tables say what the fit did (game plan 2.2), in batch and in the GUI.

- bounds_* columns hold the bounds the fit used, not the full x range or the
  typed window ends;
- fit_ok comes from the fit: an export before any fit, or of peaks edited since,
  says False;
- inputs that share a stem (a.csv and a.txt) get separate outputs;
- a rewrite of a spectrum's fit table removes the uncertainty files it does not
  write, so a folder never pairs a new fit with an old uncertainty;
- a typed export name keeps its dots: only a trailing .csv is dropped, so
  MoS2_0.5mW does not become MoS2_0;
- batch honours Use baseline: off means no baseline at all.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from batch import runner
from core import data_io, fit_api, peaks
from tests.synthetic_raman import BACKGROUND, TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

HEADLESS = os.environ.get("DISPLAY", "") == "" and os.name != "nt"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _config(out_dir, **extra):
    return {
        "peaks": [dict(p) for p in TEMPLATE],
        "source": "template",
        "solver": "modern_vp",
        "modern_vp": {"centers_in_window": True},
        "mode": "add",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "baseline_fit": "fixed",
        "noise_model": "none",
        "output_dir": str(out_dir),
        "output_base": "batch",
        "save_traces": True,
        **extra,
    }


@pytest.fixture
def spectrum(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "spec.csv"
    x, y = mos2_spectrum(3)
    write_spectrum(path, x, y)
    return path, x, y


def test_bounds_columns_hold_the_bounds_the_fit_used(tmp_path, spectrum):
    path, x, y = spectrum
    out = tmp_path / "out"
    cfg = _config(out)
    runner.run_batch([str(path)], cfg, compute_uncertainty=False)
    rows = _rows(out / "spec_fit.csv")

    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    base = np.asarray(runner.signals.als_baseline(y[mask], lam=1e5, p=0.001, niter=10, tol=0.0))
    baseline = np.interp(x, x[mask], base, left=base[0], right=base[-1])
    fit = fit_api.run_fit_consistent(
        x, y, peaks.deserialize(cfg["peaks"]), cfg, baseline, "add", mask
    )
    lo, hi = fit["bounds"]
    assert len(rows) == len(TEMPLATE)
    for i, row in enumerate(rows):
        assert float(row["bounds_center_lo"]) == pytest.approx(lo[4 * i])
        assert float(row["bounds_center_hi"]) == pytest.approx(hi[4 * i])
        assert float(row["bounds_height_lo"]) == pytest.approx(lo[4 * i + 1])
        assert float(row["bounds_height_hi"]) == pytest.approx(hi[4 * i + 1])
        assert float(row["bounds_fwhm_lo"]) == pytest.approx(lo[4 * i + 2])
        # centres were bounded to the points in the window, not to all of x
        assert WINDOW[0] <= float(row["bounds_center_lo"]) < float(row["bounds_center_hi"]) <= WINDOW[1]


def test_inputs_that_share_a_stem_get_separate_outputs(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    x, y = mos2_spectrum(4)
    write_spectrum(folder / "a.csv", x, y)
    np.savetxt(folder / "a.txt", np.column_stack([x, (y - BACKGROUND) * 0.5 + BACKGROUND]))
    out = tmp_path / "out"
    runner.run_batch([str(folder / "a.csv"), str(folder / "a.txt")], _config(out), compute_uncertainty=True,
                     unc_method="asymptotic")

    csv_rows, txt_rows = _rows(out / "a_csv_fit.csv"), _rows(out / "a_txt_fit.csv")
    assert {r["file"] for r in csv_rows} == {"a.csv"} and {r["file"] for r in txt_rows} == {"a.txt"}
    # the .txt lines are half as tall
    for rc, rt in zip(csv_rows, txt_rows):
        assert float(rt["height"]) == pytest.approx(0.5 * float(rc["height"]), rel=0.05)
    for stem in ("a_csv", "a_txt"):
        assert (out / f"{stem}_trace.csv").exists() and (out / f"{stem}_uncertainty.csv").exists()
    assert not (out / "a_fit.csv").exists()


def test_a_rewritten_fit_removes_the_uncertainty_it_does_not_write(tmp_path, spectrum):
    path, _, _ = spectrum
    out = tmp_path / "out"
    out.mkdir()
    old = ["spec_uncertainty.csv", "spec_uncertainty_wide.csv", "spec_uncertainty.txt",
           "spec_uncertainty_band.csv", "spec_band.csv"]
    for name in old:
        (out / name).write_text("from an earlier fit\n")
    runner.run_batch([str(path)], _config(out), compute_uncertainty=False)
    assert (out / "spec_fit.csv").exists()
    assert [n for n in old if (out / n).exists()] == []


def test_batch_honours_use_baseline_off(tmp_path, spectrum):
    path, x, y = spectrum
    out = tmp_path / "out"
    runner.run_batch([str(path)], _config(out, use_baseline=False), compute_uncertainty=False)
    trace = _rows(out / "spec_trace.csv")
    assert {float(r["baseline"]) for r in trace} == {0.0}
    rows = _rows(out / "spec_fit.csv")
    assert {r["use_baseline"] for r in rows} == {"False"}
    # with no baseline the peaks sit on the 505-count background: the fit is of y itself
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    fit = fit_api.run_fit_consistent(
        x, y, peaks.deserialize(_config(out)["peaks"]), _config(out), np.zeros_like(x), "add", mask
    )
    for row, pk in zip(rows, fit["peaks_out"]):
        assert float(row["height"]) == pytest.approx(pk.height, rel=1e-9)


def test_derive_export_paths_drops_only_a_csv_suffix(tmp_path):
    typed = data_io.derive_export_paths(str(tmp_path / "MoS2_0.5mW"))
    assert typed["fit"].name == "MoS2_0.5mW_fit.csv"
    assert typed["unc_band"].name == "MoS2_0.5mW_uncertainty_band.csv"
    picked = data_io.derive_export_paths(str(tmp_path / "MoS2_0.5mW.csv"))
    assert picked["trace"].name == "MoS2_0.5mW_trace.csv"


# --- the GUI ------------------------------------------------------------------

gui_only = pytest.mark.skipif(HEADLESS, reason="drives the real GUI: needs a display")


@pytest.fixture
def gui(monkeypatch):
    tk = pytest.importorskip("tkinter")
    from tests.gui_driver import Driver
    from ui import app as ui_app

    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    driver.Peak = ui_app.Peak
    try:
        yield driver
    finally:
        root.destroy()


def _open_and_seed(gui, path):
    gui.open_file(path)
    gui.set_window(*WINDOW)
    gui.app.peaks = [gui.Peak(**dict(p)) for p in TEMPLATE]
    gui.app.refresh_tree()


@gui_only
def test_fit_ok_comes_from_the_fit(gui, tmp_path, spectrum):
    path, _, _ = spectrum
    _open_and_seed(gui, path)
    gui.export(tmp_path / "before.csv")
    assert {r["fit_ok"] for r in _rows(tmp_path / "before_fit.csv")} == {"False"}

    gui.fit()
    gui.export(tmp_path / "after.csv")
    assert {r["fit_ok"] for r in _rows(tmp_path / "after_fit.csv")} == {"True"}

    gui.app.peaks[0].center += 0.5  # edited by hand after the fit
    gui.export(tmp_path / "edited.csv")
    assert {r["fit_ok"] for r in _rows(tmp_path / "edited_fit.csv")} == {"False"}


@gui_only
def test_a_typed_dotted_export_name_keeps_its_dots(gui, tmp_path, spectrum):
    path, _, _ = spectrum
    _open_and_seed(gui, path)
    gui.fit()
    # Tk's save dialog returns a dotted name without adding .csv
    gui.export(tmp_path / "MoS2_0.5mW")
    names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert "MoS2_0.5mW_fit.csv" in names and "MoS2_0.5mW_trace.csv" in names
    assert "MoS2_0.5mW_uncertainty.csv" in names
    assert not [n for n in names if n.startswith("MoS2_0_")]


@gui_only
def test_a_typed_dotted_batch_summary_name_keeps_its_dots(gui, tmp_path, spectrum):
    path, _, _ = spectrum
    _open_and_seed(gui, path)
    gui.app.batch_source.set("current")
    out = tmp_path / "out"
    out.mkdir()
    gui.run_batch(path.parent, out / "series_0.5mW")
    assert (out / "series_0.5mW_fit.csv").exists()


@gui_only
def test_an_export_removes_the_uncertainty_files_it_does_not_write(gui, tmp_path, spectrum):
    path, _, _ = spectrum
    _open_and_seed(gui, path)
    gui.fit()
    for name in ("spec_uncertainty_wide.csv", "spec_band.csv"):
        (tmp_path / name).write_text("from an earlier export\n")
    gui.app.cfg["export_unc_wide"] = False
    gui.export(tmp_path / "spec.csv")
    assert (tmp_path / "spec_uncertainty.csv").exists()
    assert not (tmp_path / "spec_uncertainty_wide.csv").exists()
    assert not (tmp_path / "spec_band.csv").exists()
