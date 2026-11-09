"""Every solver fits every height within one bound, and a peak held outside the
window cannot take the fit.

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

Now fit.bounds.pack_theta_bounds gives every height one bound, which every solver
enforces in its solve: the data's cap (height_cap, twice the largest value the
peaks are fitted to) times the most of the peak the window can show
(visible_share: 1 for a centre the window can hold, the value at the nearer
window edge for one locked outside). A peak the window cannot see is held at 0,
and flagged.
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


def test_the_height_bound_is_the_cap_times_what_the_window_shows():
    from fit.bounds import height_cap, pack_theta_bounds, visible_share

    x = np.linspace(WINDOW[0], WINDOW[1], 95)
    peaks = _seeds()
    _, (lo, hi) = pack_theta_bounds(peaks, x, {"max_height": 100.0})
    shares = [visible_share(p.center, p.fwhm, p.eta, *WINDOW) for p in peaks]
    assert shares[:2] == [1.0, 1.0]
    assert 1e-3 < shares[2] < 1e-2 and 0.4 < shares[4] < 0.6
    assert hi[1::4][:2] == pytest.approx([100.0, 100.0])
    assert hi[1::4][2] == pytest.approx(100.0 * shares[2]) and hi[1::4][4] == pytest.approx(100.0 * shares[4])
    for i in INVISIBLE:  # held at 0, as pack_theta_bounds holds a locked value
        assert lo[4 * i + 1] == 0.0 and hi[4 * i + 1] <= 1e-12
    # a free centre may sit in the window: its peak keeps the whole cap
    free = [Peak(450.0, 1.0, 5.0, 0.0)]
    assert pack_theta_bounds(free, x, {"max_height": 100.0})[1][1][1] == pytest.approx(100.0)
    # the cap is the data's, or the one given (a bootstrap refit takes its fit's)
    assert height_cap(np.array([-3.0, 2.0, 1.0])) == pytest.approx(6.0)
    assert height_cap(np.array([1.0]), {"max_height": 7.5}) == 7.5


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
