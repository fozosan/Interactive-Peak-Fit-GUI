"""Peaks a template holds outside the fit window cannot take the window's area,
and a height stopped just short of the cap is at it.

The round-3 check of #257 (ec265e6, against main e9d708f):
- Classic: templates that lock peaks far outside the window (2400H-2ZF-Rad, six
  at 126-242 cm-1) left those peaks holding most of the area. In the GUI,
  2400H-2ZF-Rad on the 5 s spectrum fell from 95.6 % of the area in the window
  to 2.4 %, its far peaks at 175-178 under a cap of 180.6; 2ZSF-SiO2-ML-2,
  good on main, was refused at the cap. Main's Classic had moved locked centres
  into the window, which round 1 stopped; round 3 dropped round 2's visible
  share, the bound that had kept such peaks small.
- Every solver's refit bound a height it had put past the cap by the cap
  itself, so a far peak's tail took the background up to just under it (Modern
  TRF 2ZF-Rad, 5 s: a peak at 188 cm-1 at 684.805 of 684.821, 97 % of the area).
  That height, 2.5e-5 short of the cap, was not flagged: the check's tolerance
  was 1e-6, and least_squares' interior steps stop short of a bound.

Now a height the cap bounds is bound by the cap times its visible share (1 for
a peak the window can hold), in every solver; Classic, whose trust region walks
such a height along the valley its tail shares with the baseline without ever
reaching the cap, bounds a peak the window cannot hold from its first pass. A
height within 1e-3 of its bound is at it, and fails the fit when its peak's
largest value in the window at that bound reaches the window's peak signal
(or the cap, if smaller): always at the cap; for these far peaks, bound at
0.05 to 0.17 % of the cap, it is a flag in bound_hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import fit_api
from core.models import pseudo_voigt
from core.peaks import Peak
from tests.synthetic_raman import BACKGROUND, MOS2_WEAK_LINES, WEAK_BACKGROUND, mos2_spectrum, mos2_x

WINDOW = (360.0, 430.0)
# 2400H-2ZF-Rad's peaks locked at 126-242 cm-1 (centre, FWHM, eta from the user's template)
FAR = [(126.12, 21.9, 0.5), (152.43, 21.5, 0.3), (177.28, 21.4, 0.5), (187.52, 21.7, 0.3),
       (228.43, 13.7, 0.3), (241.83, 7.32, 0.5)]


def _weak_spectrum():
    # E2g and A1g of the 0.20 s spectrum (9.8 and 12.5 counts) with its noise
    rng = np.random.default_rng(3)
    x = mos2_x()
    mean = np.full_like(x, WEAK_BACKGROUND) + sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in MOS2_WEAK_LINES)
    y = mean + rng.normal(0.0, np.sqrt(0.724 + 0.0451 * (mean - 502.5)))
    return x, y, (x >= WINDOW[0]) & (x <= WINDOW[1])


def _seeds():
    return [Peak(383.0, 8.0, 3.0, 0.5), Peak(408.0, 10.0, 3.0, 0.5)] + [
        Peak(c, 5.0, w, e, True, True) for c, w, e in FAR
    ]


def _in_window_share(theta):
    xx = np.linspace(-2000.0, 3000.0, 200001)
    areas = np.array([pseudo_voigt(xx, h, c, w, e).sum() for c, h, w, e in np.asarray(theta, float).reshape(-1, 4)])
    return areas[:2].sum() / areas.sum()


def _check_far_peaks_kept_small(res, x, mask):
    from fit.bounds import visible_share

    theta = np.asarray(res["theta"], float)
    hi = np.asarray(res["bounds"][1], float)
    assert res["fit_ok"], res["message"]
    assert res["rmse"] < 0.9  # the noise: E2g and A1g kept
    assert np.asarray(theta[[1, 5]]) == pytest.approx([9.8, 12.5], rel=0.25)
    # round 3: 0.1 % to 7 % of the area in the window; main's Classic keeps 47 %
    # on the 0.20 s spectrum, its far peaks moved to the window's edge
    assert _in_window_share(theta) > 0.4
    for i, (c, w, e) in enumerate(FAR, start=2):
        share = visible_share(c, w, e, float(x[mask].min()), float(x[mask].max()))
        assert theta[4 * i + 1] <= res["height_cap"] * share * (1 + 1e-9)
        assert hi[4 * i + 1] == pytest.approx(res["height_cap"] * share)
        if theta[4 * i + 1] >= hi[4 * i + 1] * (1 - 1e-3):  # at its bound: flagged, not failed
            assert f"height{i + 1}" in res["bound_hit"]
    assert not res["height_at_cap"]


@pytest.mark.parametrize("baseline_fit", ["joint", "fixed"])
def test_classic_keeps_a_template_s_far_peaks_from_taking_the_window(baseline_fit):
    # Round 3: refused (the far peaks at the cap, 529 with a joint baseline and 26
    # with a fixed one) with 0.1 to 7 % of the area in the window.
    x, y, mask = _weak_spectrum()
    cfg = {"solver": "classic", "classic": {"maxfev": 20000}, "baseline_fit": baseline_fit}
    res = fit_api.run_fit_consistent(x, y, _seeds(), cfg, np.full_like(x, WEAK_BACKGROUND), "add", mask)
    _check_far_peaks_kept_small(res, x, mask)


@pytest.mark.parametrize("solver", ["modern_vp", "lmfit_vp"])
def test_a_far_peak_a_fit_puts_past_the_cap_is_refit_within_its_share(solver):
    # NNLS runs a far peak past the cap with a joint baseline; round 3's refit
    # bound it by the cap itself, and the next one ran away in its place, until
    # all six ended at the cap: refused, 0.1 % in the window.
    x, y, mask = _weak_spectrum()
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "baseline_fit": "joint"}
    res = fit_api.run_fit_consistent(x, y, _seeds(), cfg, np.full_like(x, WEAK_BACKGROUND), "add", mask)
    _check_far_peaks_kept_small(res, x, mask)


@pytest.mark.parametrize("solver", ["classic", "modern_trf", "modern_vp", "lmfit_vp"])
def test_a_height_just_short_of_the_cap_is_at_it(solver, monkeypatch):
    # least_squares' interior steps stop short of an active bound: Modern TRF left
    # a height 2.5e-5 under the cap, which a 1e-6 tolerance passed as free.
    x, y = mos2_spectrum(7)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    seeds = [Peak(383.0, 60.0, 3.0, 0.5), Peak(408.0, 80.0, 3.0, 0.5)]
    run = fit_api.orchestrator.run_fit_with_fallbacks

    def short_of_the_cap(*args, **kwargs):
        res = run(*args, **kwargs)
        theta = np.asarray(res.theta, float)
        theta[5] = min(theta[5], 80.0 * (1 - 2.5e-5))  # A1g, which the cap holds at 80
        res.theta = theta
        return res

    monkeypatch.setattr(fit_api.orchestrator, "run_fit_with_fallbacks", short_of_the_cap)
    cfg = {"solver": solver, solver: {"maxfev": 20000, "max_height": 80.0}}
    res = fit_api.run_fit_consistent(x, y, seeds, cfg, np.full_like(x, BACKGROUND), "add", mask)
    assert np.asarray(res["theta"], float)[5] == pytest.approx(80.0 * (1 - 2.5e-5))
    assert not res["fit_ok"]
    assert res["height_at_cap"] == ["height2"]
    assert "height2" in res["bound_hit"]
