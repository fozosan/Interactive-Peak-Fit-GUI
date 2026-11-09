"""One model, peaks plus a baseline (game plan 1.6, decision D2).

``run_fit_consistent`` takes ``cfg["baseline_fit"]``:

- ``"fixed"``, the core default and the interactive path, holds the given (ALS or
  polynomial) baseline fixed, as before;
- ``"joint"`` fits the baseline with the peaks: Legendre columns of order
  ``cfg["baseline_order"]`` (default 3) over the fit window, free linear
  parameters solved beside the peak heights. The given baseline only seeds the
  start (``reheight``).

The fixed baseline is the special case with no free columns. Every solver fits
the same model: at the weighted optimum the gradient J^T W r vanishes over the
peak parameters and the baseline coefficients alike.
"""

import numpy as np
import pytest

from core import baseline_model as BM
from core import fit_api
from core import noise_model as NM
from core.models import pseudo_voigt
from core.peaks import Peak

LINES = ((383.0, 60.0, 2.3, 0.5), (408.0, 80.0, 2.8, 0.5))
COEF = np.array([505.0, 1.0, -0.4, 0.3])  # a cubic background, inside the model
SOLVERS = ["modern_vp", "modern_trf", "classic", "lmfit_vp"]


def _x():
    return np.linspace(320.0, 460.0, 191)


def _mask(x):
    return (x >= 340.0) & (x <= 440.0)


def _signal(x, theta):
    return sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in np.asarray(theta, float).reshape(-1, 4))


def _truth_theta():
    return np.array([v for line in LINES for v in line], float)


def _background(x, mask):
    return BM.JointBaseline.over(x[mask], 3).basis(x) @ COEF


def _spectrum(seed=None, scale=1.0):
    x = _x()
    mask = _mask(x)
    mu = _background(x, mask) + _signal(x, _truth_theta())
    if seed is None:
        return x, mu, mask
    sigma = np.sqrt(0.8 + 0.045 * _signal(x, _truth_theta()))
    rng = np.random.default_rng(seed)
    return x, mu + scale * sigma * rng.standard_normal(x.size), mask


def _seeds():
    return [Peak(383.4, 50.0, 3.0, 0.5), Peak(407.6, 70.0, 3.0, 0.5)]


def _cfg(solver, **extra):
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "baseline_fit": "joint"}
    cfg.update(extra)
    return cfg


def _flat(x):
    return np.full_like(x, 500.0)


def test_the_core_default_holds_the_given_baseline_fixed():
    x, y, mask = _spectrum(31)
    base = _flat(x)
    plain = fit_api.run_fit_consistent(x, y, _seeds(), {"solver": "modern_vp"}, base, "add", mask)
    fixed = fit_api.run_fit_consistent(
        x, y, _seeds(), {"solver": "modern_vp", "baseline_fit": "fixed"}, base, "add", mask
    )
    assert plain["baseline_fit"] == "fixed" and plain["baseline_order"] is None
    assert plain["baseline_coef"] is None and plain["joint_baseline"] is None
    np.testing.assert_array_equal(plain["theta"], fixed["theta"])
    np.testing.assert_array_equal(plain["baseline"], base)


def test_unknown_baseline_fit_is_a_setting_error():
    x, y, mask = _spectrum(32)
    with pytest.raises(ValueError, match="baseline_fit"):
        fit_api.run_fit_consistent(x, y, _seeds(), _cfg("modern_vp", baseline_fit="loess"), None, "add", mask)


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_recovers_peaks_and_a_curved_baseline(solver):
    x, y, mask = _spectrum()
    # The given baseline is a poor flat guess; joint mode does not hold it.
    res = fit_api.run_fit_consistent(x, y, _seeds(), _cfg(solver), _flat(x), "add", mask)
    assert res["fit_ok"], res["message"]
    assert res["baseline_fit"] == "joint" and res["baseline_order"] == 3
    np.testing.assert_allclose(res["theta"], _truth_theta(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(res["baseline_coef"], COEF, rtol=1e-5, atol=1e-4)
    jb = res["joint_baseline"]
    assert (jb.lo, jb.hi) == (x[mask][0], x[mask][-1])
    # the result's baseline is the fitted one, constant beyond the window
    np.testing.assert_allclose(res["baseline"][mask], _background(x, mask)[mask], atol=1e-4)
    assert res["baseline"][0] == pytest.approx(res["baseline"][mask][0])
    assert res["rmse"] < 1e-4


def _gradient(xf, yf, theta, coef, sigma):
    """J^T W r over the free parameters: centre, height, FWHM of each peak and the
    baseline coefficients, each column scaled so the entries compare."""
    theta = np.asarray(theta, float)
    B = BM.JointBaseline.over(xf, coef.size - 1).basis(xf)

    def resid(th, c):
        return (_signal(xf, th) + B @ c - yf) / sigma

    r0 = resid(theta, coef)
    cols = []
    for i in [i for i in range(theta.size) if i % 4 != 3]:
        th = theta.copy()
        step = 1e-6 * max(1.0, abs(theta[i]))
        th[i] += step
        cols.append((resid(th, coef) - r0) / step)
    cols.extend(B.T / sigma)
    J = np.column_stack(cols)
    return (J.T @ r0) / np.linalg.norm(J, axis=0) / np.linalg.norm(r0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_sits_at_the_weighted_joint_optimum(solver):
    x, y, mask = _spectrum(33)
    xf, yf = x[mask], y[mask]
    sigma_win = np.sqrt(0.8 + 0.045 * _signal(xf, _truth_theta()))
    sigma_full = np.ones_like(x)
    sigma_full[mask] = sigma_win
    res = fit_api.run_fit_consistent(
        x, y, _seeds(), _cfg(solver, noise_sigma=sigma_full), _flat(x), "add", mask
    )
    assert res["fit_ok"], res["message"]
    grad = _gradient(xf, yf, res["theta"], res["baseline_coef"], sigma_win)
    assert np.max(np.abs(grad)) < 1e-4, grad
    # the unweighted joint fit is not the weighted optimum
    plain = fit_api.run_fit_consistent(x, y, _seeds(), _cfg(solver), _flat(x), "add", mask)
    assert np.max(np.abs(_gradient(xf, yf, plain["theta"], plain["baseline_coef"], sigma_win))) > 1e-3


@pytest.mark.parametrize("solver", SOLVERS)
def test_centred_columns_solve_thin_data_far_from_the_origin(solver):
    # Prior art (fit-affines-on-centered-coordinates): a solve on raw coordinates
    # survives many points and fails on few. Here 15 points, centres and widths
    # locked so the fit is the linear solve itself, near the origin and 1e4 cm-1
    # out: both recover the heights and the cubic exactly, because the columns
    # are centred and scaled (cubic monomials of raw x have cond 1e12 here and
    # 5e20 out there).
    x = np.linspace(370.0, 420.0, 15)
    mask = np.ones_like(x, bool)
    lines = ((383.0, 60.0, 8.0, 0.5), (408.0, 80.0, 10.0, 0.5))  # wider than 2 points
    truth = np.array([v for line in lines for v in line], float)
    coef = np.array([505.0, 1.0, -0.4, 0.3])
    y = _signal(x, truth) + BM.JointBaseline.over(x, 3).basis(x) @ coef
    for offset in (0.0, 1.0e4):
        seeds = [Peak(c + offset, 10.0, w, e, lock_center=True, lock_width=True) for c, _h, w, e in lines]
        res = fit_api.run_fit_consistent(x + offset, y, seeds, _cfg(solver), None, "add", mask)
        assert res["fit_ok"], res["message"]
        np.testing.assert_allclose(res["theta"][1::4], truth[1::4], rtol=1e-7)
        np.testing.assert_allclose(res["baseline_coef"], coef, rtol=1e-7, atol=1e-7)


def test_order_zero_is_a_free_offset():
    x, y, mask = _spectrum()
    y = y - _background(x, mask) + 503.0  # a flat background
    res = fit_api.run_fit_consistent(x, y, _seeds(), _cfg("modern_vp", baseline_order=0), None, "add", mask)
    assert res["baseline_order"] == 0
    np.testing.assert_allclose(res["baseline_coef"], [503.0], atol=1e-6)
    np.testing.assert_allclose(res["theta"], _truth_theta(), rtol=1e-6, atol=1e-6)


def test_joint_counts_the_coefficients_as_free_parameters():
    x, y, mask = _spectrum(35)
    res = fit_api.run_fit_consistent(
        x, y, _seeds(), _cfg("modern_vp"), None, "add", mask, return_jacobian=True
    )
    n = int(mask.sum())
    assert res["dof"] == n - 6 - 4
    # too few points for 6 peak parameters plus 4 coefficients
    few = np.zeros_like(mask)
    few[np.flatnonzero(mask)[60:70]] = True
    skip = fit_api.run_fit_consistent(x, y, _seeds(), _cfg("modern_vp"), None, "add", few)
    assert skip["skip_reason"] == "too_few_points"
    assert "10 free parameters" in skip["message"]


def test_the_noise_model_sees_the_baseline_columns(monkeypatch):
    # The fit shrinks its residuals through the baseline columns as well, so the
    # noise model's leverage carries them: its Jacobian has 6 + 4 columns.
    seen = {}
    real = NM.estimate_noise_model

    def spy(residual, signal, jacobian=None, **kw):
        seen["shape"] = np.asarray(jacobian).shape
        seen["signal_min"] = float(np.min(signal))
        return real(residual, signal, jacobian, **kw)

    monkeypatch.setattr(fit_api.noise_model, "estimate_noise_model", spy)
    x, y, mask = _spectrum(36)
    res = fit_api.run_fit_consistent(
        x, y, _seeds(), _cfg("modern_vp", noise_model="per_spectrum"), None, "add", mask
    )
    assert seen["shape"] == (int(mask.sum()), 10)
    # S_i is the signal above the fitted baseline: peaks only, never the baseline level
    assert seen["signal_min"] < 1.0
    assert res["noise_model"] is not None and res["noise_sigma"].size == int(mask.sum())


def test_subtract_mode_fits_the_same_joint_model():
    x, y, mask = _spectrum(37)
    add = fit_api.run_fit_consistent(x, y, _seeds(), _cfg("modern_vp"), _flat(x), "add", mask)
    sub = fit_api.run_fit_consistent(x, y, _seeds(), _cfg("modern_vp"), _flat(x), "subtract", mask)
    np.testing.assert_allclose(sub["theta"], add["theta"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sub["baseline_coef"], add["baseline_coef"], rtol=1e-9, atol=1e-9)
