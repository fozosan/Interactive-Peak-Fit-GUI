"""No solver result is clipped after the solve; a bound hit is flagged.

The 2026-11-08 review of #253-#255: the Classic solver kept its own bounds (FWHM
unbounded above), and run_fit_consistent clipped its result onto the fit's bounds
afterwards (FWHM at most half the window). On the 5 s MoS2 spectrum, window
440-470, one peak at 452: the FWHM came out 14.629 instead of 15.547, the height
and centre stayed those of the unclipped fit, fit_ok was True, and nothing said
so. The other solvers fit within the bounds, and were pinned at 14.629 just as
silently in the GUI.

Now Classic fits within the bounds every solver reports (fit.bounds.
pack_theta_bounds), nothing is clipped afterwards, and a parameter on a bound is
named in _fit.csv (bound_hit, with the FWHM's upper bound in bounds_fwhm_hi) and
in the GUI's status line, as in batch's status table.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from core import fit_api, pipeline
from core.models import pseudo_voigt
from core.peaks import Peak
import fit as fit_pkg  # after core, which imports fit in the order fit needs
from tests.gui_driver import Driver, new_root
from tests.synthetic_raman import BACKGROUND, mos2_x, write_spectrum

# A broad band, as the 2LA(M) region of the MoS2 spectra: FWHM 16 in a 30 cm-1
# window, where the FWHM bound is half the window, about 14.6.
WINDOW = (440.0, 470.0)
BAND = dict(center=452.0, height=20.0, fwhm=16.0, eta=0.5)


def _spectrum():
    x = mos2_x()
    rng = np.random.default_rng(3)
    y = BACKGROUND + pseudo_voigt(x, BAND["height"], BAND["center"], BAND["fwhm"], BAND["eta"])
    return x, y + 0.6 * rng.standard_normal(x.size)


def _settings(solver):
    opts = {"maxfev": 20000}
    opts.update({"bound_centers_to_window": True} if solver == "classic" else {"centers_in_window": True})
    return {"solver": solver, solver: opts, "baseline": {"method": "als"}, "fit_xmin": WINDOW[0],
            "fit_xmax": WINDOW[1]}


def _fit(solver):
    x, y = _spectrum()
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(452.0, 15.0, 8.0, 0.5)]
    res = fit_api.run_fit_consistent(x, y, seeds, _settings(solver), np.full_like(x, BACKGROUND), "add", mask)
    return x, y, res


def test_classic_finds_the_best_fit_within_the_bounds():
    _, _, classic = _fit("classic")
    _, _, trf = _fit("modern_trf")
    lo, hi = classic["bounds"]
    assert classic["fit_ok"]
    assert classic["theta"][2] == pytest.approx(hi[2], rel=1e-9)  # on the FWHM bound
    assert classic["bound_hit"] == ["fwhm1"]
    # the constrained optimum, as the Modern solver finds it: not the free fit clipped
    assert classic["rmse"] == pytest.approx(trf["rmse"], rel=1e-6)
    assert np.allclose(classic["theta"], trf["theta"], rtol=1e-4)


def test_fit_csv_names_the_parameter_on_a_bound(tmp_path):
    x, y, res = _fit("classic")
    pipeline.write_spectrum(tmp_path / "band", tmp_path / "band.csv", x=x, y_raw=y, fit=res,
                            settings=_settings("classic"), trace=False)
    with open(tmp_path / "band_fit.csv", newline="", encoding="utf-8") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["bound_hit"] == "fwhm"
    assert float(row["bounds_fwhm_hi"]) == pytest.approx(res["bounds"][1][2])
    assert float(row["fwhm"]) == pytest.approx(float(row["bounds_fwhm_hi"]))


def test_a_value_outside_the_bounds_is_reported_not_clipped(monkeypatch):
    real = fit_pkg._SOLVERS["classic"]
    pushed = {}

    def past_the_bound(x, y, peaks, mode, baseline, opts):
        out = real(x, y, peaks, mode, baseline, opts)
        width = 1.05 * 0.5 * (float(np.max(x)) - float(np.min(x)))
        out["theta"] = np.asarray(out["theta"], float).copy()
        out["theta"][2] = width
        out["peaks"][0].fwhm = width
        pushed["fwhm"] = width
        return out

    monkeypatch.setitem(fit_pkg._SOLVERS, "classic", past_the_bound)
    _, _, res = _fit("classic")
    assert res["theta"][2] == pytest.approx(pushed["fwhm"])  # what the solver returned
    assert not res["fit_ok"]
    assert "outside the fit's bounds: fwhm1" in res["message"]


@pytest.mark.skipif(os.environ.get("DISPLAY", "") == "" and os.name != "nt",
                    reason="drives the real GUI: needs a display (run under xvfb-run)")
def test_the_gui_says_when_a_fit_ends_on_a_bound(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    from ui import app as ui_app

    root = new_root(monkeypatch)
    try:
        app = ui_app.PeakFitApp(root)
        gui = Driver(app, monkeypatch)
        path = tmp_path / "band.csv"
        x, y = _spectrum()
        write_spectrum(path, x, y - BACKGROUND)  # the band alone: no baseline to fit
        gui.open_file(path)
        app.use_baseline.set(False)
        app.solver_choice.set("classic")
        app._on_solver_change()
        gui.set_window(*WINDOW)
        app.peaks = [ui_app.Peak(452.0, 15.0, 8.0, 0.5)]
        app.refresh_tree()
        gui.fit()
        hit = [line for line in app._log_buffer if "Fit at bound:" in line]
        assert hit and "fwhm1" in hit[-1], app._log_buffer[-5:]
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
