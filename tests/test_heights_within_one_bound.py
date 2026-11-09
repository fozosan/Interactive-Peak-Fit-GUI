"""Every height within one cap, which bounds only the heights a fit puts above
it, and fails a fit that leaves one at it; a peak held outside the window cannot
take the fit.

The 2026-11-09 check of #257. A template's peaks locked outside the fit window
reach the data with their tails only, which cannot fix their heights, and no
solver bounded them (make_bounds' 1e12 stands in for "no bound"):
- Classic drove them to 4e7 in the GUI (trace spike at 180 cm-1, the window's
  peaks down to 2e-5 % of the area, A1g lost on 2400H-2ZF-Rad), fit_ok True;
- Modern VP drove a near-Gaussian one at 464 cm-1 to 4.6e15 in the GUI, 5.6e16
  in batch: past the 1e12 that run_fit_consistent checked it against, so a fit
  that was fine came out "not converged";
- Modern TRF kept a height the window cannot see at its start value (21 % of the
  area in the GUI), and drove one to 1.5e8 in batch; LMFIT-VP's NNLS put them at
  0 in the GUI and at 2.5e6 in batch (2400H-2ZF-Rad on the 0.20 s spectrum).

Round 2 bounded every height by twice the peak signal from the start. Its
re-check found 13 sweep fits main got right made silently wrong (Modern VP
2ZF-Rad-3 on the 5 s spectrum: RMSE 0.79 to 12.6, a height at the cap of 181
where main's fit has 258), the default path (auto seeds, LMFIT-VP, joint
baseline) not converged, and the Modern TRF sweep 7x slower:
- with a joint baseline, which can dip below the data, twice the signal above
  the computed baseline is no bound on a broad peak: fit.bounds.height_cap is
  now the data above zero counts plus that signal (595 + 91 there);
- a bound the fit stays below still changed it: least_squares scales every step
  by the distance to the bound, and NNLS's path met it where heights pass it on
  the way. run_fit_consistent now bounds a height only once the fit puts it
  above the cap, and runs the fit again from the same start;
- a height that ends at the cap fails the fit, by name (batch status
  height_at_cap): the cap, not the data, set it.
A peak the window cannot see (visible_share below sqrt(eps)) is held at 0 in
every pass, and flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import fit_api
from core.models import pseudo_voigt
from core.peaks import Peak
from tests.synthetic_raman import BACKGROUND, mos2_spectrum

WINDOW = (360.0, 430.0)
SOLVERS = ("classic", "modern_trf", "modern_vp", "lmfit_vp")
# CVD-MoS2-Yale, the user's template, on the 5 s spectrum: E2g and A1g free, and
# its peaks outside 360-430 locked, with their shapes and heights scaled to the
# spectrum (A1g 1.99e4 there, 92 here). 177.5 and 463.7 are Gaussians to 1e-40.
FAR = [
    (150.16, 0.77, 23.4, 1.0),  # a Lorentzian whose tail the window shows at 0.3 %
    (177.52, 0.90, 13.6, 6.34e-41),  # seen at 1e-43: nothing
    (437.83, 3.6, 16.5, 1.0),  # 8 cm-1 past the window: half of it shows
    (463.65, 5.0, 9.25, 5.55e-39),  # nothing
]
INVISIBLE = (3, 5)  # the two the window cannot see, as indices into _seeds()


def _seeds():
    seeds = [Peak(383.0, 60.0, 3.0, 0.5), Peak(408.0, 80.0, 3.0, 0.5)]
    seeds += [Peak(c, h, w, e, True, True) for c, h, w, e in FAR]
    return seeds


def _fit(solver, baseline_fit="fixed"):
    x, y = mos2_spectrum(7)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "baseline_fit": baseline_fit}
    res = fit_api.run_fit_consistent(x, y, _seeds(), cfg, np.full_like(x, BACKGROUND), "add", mask)
    return x, y, mask, res


def _areas(theta):
    return np.array([pseudo_voigt(np.linspace(-2000, 3000, 200001), h, c, w, e).sum() for c, h, w, e in
                     np.asarray(theta, float).reshape(-1, 4)])


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_peak_held_outside_the_window_cannot_take_the_fit(solver):
    x, y, mask, res = _fit(solver)
    theta = np.asarray(res["theta"], float)
    lo, hi = (np.asarray(b, float) for b in res["bounds"])
    heights = theta[1::4]

    assert res["fit_ok"], res["message"]
    assert not res["outside_bounds"]
    assert np.all(heights <= hi[1::4] * (1 + 1e-9)) and np.all(heights >= lo[1::4])
    # the window's two lines keep their heights and nearly all of the area
    assert heights[:2] == pytest.approx([68.7, 92.3], rel=0.06)
    area = _areas(theta)
    assert area[:2].sum() / area.sum() > 0.9, area / area.sum()
    # no spike outside the window: no peak above the data's cap anywhere
    cap = res["height_cap"]
    assert cap == pytest.approx(2 * np.max(np.abs(y[mask] - BACKGROUND)))
    assert np.all(heights <= cap)
    # the peaks the window cannot see have no height, and say so
    for i in INVISIBLE:
        assert heights[i] == 0.0
        assert f"height{i + 1}" in res["bound_hit"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_batch_fit_with_its_baseline_keeps_every_height_in_bounds(solver):
    # Batch fits the baseline with the peaks (baseline_fit "joint"): Modern VP's
    # 464 cm-1 peak reached 5.6e16 there and the fit was "not converged".
    _, _, _, res = _fit(solver, baseline_fit="joint")
    theta = np.asarray(res["theta"], float)
    hi = np.asarray(res["bounds"][1], float)
    assert not res["outside_bounds"]
    assert res["fit_ok"], res["message"]
    assert np.all(theta[1::4] <= hi[1::4] * (1 + 1e-9))
    assert np.all(theta[1::4] <= res["height_cap"])


def test_the_cap_bounds_every_height_it_is_given_to_times_its_share_and_none_other():
    from fit.bounds import NO_BOUND, pack_theta_bounds, visible_share

    x = np.linspace(WINDOW[0], WINDOW[1], 95)
    peaks = _seeds()
    shares = [visible_share(p.center, p.fwhm, p.eta, *WINDOW) for p in peaks]
    assert shares[:2] == [1.0, 1.0]
    assert 1e-3 < shares[2] < 1e-2 and 0.4 < shares[4] < 0.6
    assert shares[3] < 1e-40 and shares[5] < 1e-15
    visible = [i for i in range(len(peaks)) if i not in INVISIBLE]
    # a height the cap bounds takes it times the share of its peak the window
    # shows: a peak locked outside the window cannot take more than its tail
    reported = []
    _, (lo, hi) = pack_theta_bounds(peaks, x, {"max_height": 100.0}, shares=reported)
    assert reported == pytest.approx(shares)
    assert hi[1::4][visible] == pytest.approx([100.0 * shares[i] for i in visible])
    for i in INVISIBLE:  # held at 0, as pack_theta_bounds holds a locked value
        assert lo[4 * i + 1] == 0.0 and hi[4 * i + 1] <= 1e-12
    # a height the cap does not bound keeps make_bounds' stand-in for none; a
    # peak the window cannot see stays held
    capped = np.zeros(len(peaks), bool)
    capped[1] = True
    _, (lo, hi) = pack_theta_bounds(peaks, x, {"max_height": 100.0, "capped": capped})
    assert hi[1::4][1] == pytest.approx(100.0)
    assert all(hi[4 * i + 1] == NO_BOUND for i in visible if i != 1)
    for i in INVISIBLE:
        assert hi[4 * i + 1] <= 1e-12
    # a free centre may sit in the window: its peak keeps the whole cap
    free = [Peak(450.0, 1.0, 5.0, 0.0)]
    assert pack_theta_bounds(free, x, {"max_height": 100.0})[1][1][1] == pytest.approx(100.0)


def test_the_cap_is_the_data_above_the_lowest_the_baseline_can_lie_plus_the_signal():
    from fit.bounds import height_cap

    # a fixed baseline lies where it is: twice the peak signal (|y - baseline|)
    assert height_cap(np.array([-3.0, 2.0, 1.0])) == pytest.approx(6.0)
    # a joint one can dip below the data, not below zero counts: max(y) + signal
    y = np.array([505.0, 595.0, 510.0])
    assert height_cap(y - 504.0, y_data=y) == pytest.approx(595.0 + 91.0)
    # data that go below zero: their minimum is the floor
    y = np.array([-20.0, 70.0, -5.0])
    assert height_cap(y - (-20.0), y_data=y) == pytest.approx(90.0 + 90.0)
    # the one given (a bootstrap refit takes its fit's)
    assert height_cap(np.array([1.0]), {"max_height": 7.5}) == 7.5
    assert height_cap(y, {"max_height": 7.5}, y_data=y) == 7.5


def _broad_bump():
    # E2g and A1g on a broad peak (250 counts, FWHM 30) over a curved baseline:
    # the shape that a computed baseline follows, so twice the signal above it
    # (round 2's cap, 178 here) cuts the broad peak, as on the 5 s spectrum,
    # where main's broad peaks reach 258 and 535 under a 181 cap.
    from tests.synthetic_raman import MOS2_LINES, mos2_x

    x = mos2_x()
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    curve = BACKGROUND - 100.0 * ((x - 395.0) / 35.0) ** 2
    broad = pseudo_voigt(x, 250.0, 395.0, 30.0, 0.5)
    lines = sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in MOS2_LINES)
    seeds = [Peak(383.0, 60.0, 3.0, 0.5), Peak(408.0, 80.0, 3.0, 0.5), Peak(395.0, 100.0, 25.0, 0.5)]
    return x, curve + broad + lines, mask, curve + broad, seeds


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_broad_peak_under_the_lines_keeps_its_height_with_a_joint_baseline(solver):
    x, y, mask, computed, seeds = _broad_bump()
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "baseline_fit": "joint"}
    res = fit_api.run_fit_consistent(x, y, seeds, cfg, computed, "add", mask)
    signal = np.max(np.abs(y[mask] - computed[mask]))
    assert 2 * signal < 200.0  # round 2's cap, below the broad peak
    assert res["fit_ok"], res["message"]
    assert res["height_cap"] == pytest.approx(np.max(y[mask]) + signal)
    assert np.asarray(res["theta"], float)[1::4] == pytest.approx([68.7, 92.3, 250.0], rel=0.02)
    assert res["rmse"] < 0.5
    assert not res["height_at_cap"]


@pytest.mark.parametrize("baseline_fit", ["fixed", "joint"])
@pytest.mark.parametrize("solver", SOLVERS)
def test_a_cap_the_fit_stays_below_changes_nothing(solver, baseline_fit):
    # The same fit, to the last bit, as with no cap: least_squares scales its
    # steps by the distance to a bound, so a finite bound changes the path and
    # the cost of a fit it never touches (Modern TRF's sweep ran 7x longer).
    x, y = mos2_spectrum(7)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(383.0, 60.0, 3.0, 0.5), Peak(408.0, 80.0, 3.0, 0.5)]
    base = np.full_like(x, BACKGROUND)
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "baseline_fit": baseline_fit}
    capped = fit_api.run_fit_consistent(x, y, seeds, cfg, base, "add", mask)
    none = fit_api.run_fit_consistent(
        x, y, seeds, {**cfg, solver: {"maxfev": 20000, "max_height": 1e12}}, base, "add", mask
    )
    assert capped["fit_ok"] and capped["height_cap"] < 1e3
    assert np.array_equal(np.asarray(capped["theta"]), np.asarray(none["theta"]))
    assert capped["rmse"] == none["rmse"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_height_that_ends_at_the_cap_fails_the_fit_by_name(solver):
    # A1g is 92 counts; a cap of 80 leaves it there, set by the cap and not by
    # the data: no result, and it says which height.
    x, y = mos2_spectrum(7)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(383.0, 60.0, 3.0, 0.5), Peak(408.0, 80.0, 3.0, 0.5)]
    cfg = {"solver": solver, solver: {"maxfev": 20000, "max_height": 80.0}}
    res = fit_api.run_fit_consistent(x, y, seeds, cfg, np.full_like(x, BACKGROUND), "add", mask)
    heights = np.asarray(res["theta"], float)[1::4]
    assert heights[1] == pytest.approx(80.0)
    assert heights[0] < 75.0
    assert not res["fit_ok"]
    assert res["height_at_cap"] == ["height2"]
    assert "height2" in res["bound_hit"]
    assert "height2 ended at the cap" in res["message"]


def test_batch_says_height_at_cap_and_computes_no_uncertainty(tmp_path):
    import csv

    from batch import runner
    from tests.test_batch_file_isolation import _config

    x, y = mos2_spectrum(7)
    src = tmp_path / "in"
    src.mkdir()
    np.savetxt(src / "spec.csv", np.column_stack([x, y]), delimiter=",")
    solver = "modern_vp"
    out = tmp_path / "out"
    seeds = [{"center": c, "height": h, "fwhm": 3.0, "eta": 0.5, "lock_center": False, "lock_width": False}
             for c, h in ((383.0, 60.0), (408.0, 80.0))]
    cfg = {**_config(out, solver), "peaks": seeds, solver: {"maxfev": 20000, "max_height": 80.0},
           "fit_xmin": WINDOW[0], "fit_xmax": WINDOW[1], "unc_method": "asymptotic"}
    runner.run_batch([str(src / "spec.csv")], cfg, compute_uncertainty=True, unc_method="asymptotic")
    rows = list(csv.DictReader(open(next(out.glob("*_files.csv")), newline="", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "height_at_cap"
    assert row["fit_ok"] == "False"
    assert "height2 ended at the cap" in row["message"]
    assert row["unc_status"] == "skipped" and row["unc_message"] == "a height ended at its cap"
    assert not list(out.glob("*uncertainty*"))


def test_a_bootstrap_refits_within_the_bounds_of_its_fit(tmp_path, monkeypatch):
    # The refits take the fit's cap (result["height_cap"]), not one from each
    # resampled spectrum, so each draw is bounded as the fit was; a held height is
    # no parameter of the fit, and stays at 0.
    from core import pipeline
    from tests.test_batch_file_isolation import _config

    solver = "modern_vp"
    x, y = mos2_spectrum(7)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    settings = {**_config(tmp_path, solver), solver: {"maxfev": 20000}, "bootstrap_n": 12,
                "perf_unc_workers": 1, "fit_xmin": WINDOW[0], "fit_xmax": WINDOW[1]}
    fit = fit_api.run_fit_consistent(x, y, _seeds(), settings, np.full_like(x, BACKGROUND), "add", mask,
                                     return_jacobian=True, return_predictors=True)
    assert fit["fit_ok"]
    for i in INVISIBLE:
        assert fit["locked_mask"][4 * i + 1]

    caps = []
    refit = fit_api.run_fit_consistent

    def recording(*args, **kwargs):
        cfg = args[3]
        caps.append(cfg[cfg["solver"]].get("max_height"))
        return refit(*args, **kwargs)

    monkeypatch.setattr(fit_api, "run_fit_consistent", recording)
    res = pipeline.compute_uncertainty(fit, x, y, mask, settings, "bootstrap")
    diag = res.get("diagnostics") if isinstance(res, dict) else getattr(res, "diagnostics", {})
    assert int(diag.get("n_fail") or 0) == 0, diag.get("refit_errors")
    assert len(caps) == 12 and set(caps) == {fit["height_cap"]}
    norm = pipeline.finish_uncertainty(res, fit, settings, "bootstrap")
    for i in INVISIBLE:
        assert float(norm["stats"][i]["height"]["est"]) == 0.0
