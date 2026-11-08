"""The fit is weighted by the per-spectrum noise model (game plan 1.5, decision D1).

``run_fit_consistent`` takes ``cfg["noise_model"]``:

- ``"none"``, the core default, fits every point with the same weight, as before;
- ``"per_spectrum"`` fits once, estimates var_i = s0^2 + g * S_i from that fit's
  residuals, and refits with weights 1 / sigma_i.

``cfg["noise_sigma"]`` instead fixes sigma_i (the bootstrap refits use it). The
result carries ``noise_model`` (s0, g, n, method) and ``noise_sigma`` on the fit
window. A weighted least-squares fit is a stationary point of the weighted
objective, so the gradient J^T W r vanishes there and the unweighted one does not.
"""

import numpy as np
import pytest

from core import fit_api
from core.models import pseudo_voigt
from core.peaks import Peak
from tests.synthetic_raman import BACKGROUND, NOISE_FLOOR, NOISE_GAIN, BIAS, TEMPLATE, WINDOW, mos2_spectrum


def _window(seed=21):
    x, y = mos2_spectrum(seed)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    return x, y, mask, np.full_like(x, BACKGROUND)


def _seeds():
    return [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]


def _signal(x, theta):
    return sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in np.asarray(theta, float).reshape(-1, 4))


def _gradient(x, y, base, theta, sigma):
    """J^T W r over the free parameters (centre, height, FWHM of each peak)."""
    theta = np.asarray(theta, float)
    free = [i for i in range(theta.size) if i % 4 != 3]
    r0 = (_signal(x, theta) + base - y) / sigma
    J = []
    for i in free:
        step = 1e-6 * max(1.0, abs(theta[i]))
        th = theta.copy()
        th[i] += step
        J.append(((_signal(x, th) + base - y) / sigma - r0) / step)
    J = np.column_stack(J)
    # scale each column so the entries compare across parameters
    return (J.T @ r0) / np.linalg.norm(J, axis=0) / np.linalg.norm(r0)


def _true_sigma(x, theta_like):
    return np.sqrt(NOISE_FLOOR + NOISE_GAIN * np.maximum(BACKGROUND - BIAS + _signal(x, theta_like), 0.0))


def test_the_core_default_is_unweighted():
    x, y, mask, base = _window()
    cfg = {"solver": "modern_vp"}
    plain = fit_api.run_fit_consistent(x, y, _seeds(), cfg, base, "add", mask)
    off = fit_api.run_fit_consistent(x, y, _seeds(), {**cfg, "noise_model": "none"}, base, "add", mask)
    assert plain["noise_model"] is None and plain["noise_sigma"] is None
    np.testing.assert_array_equal(plain["theta"], off["theta"])


def test_unknown_noise_model_is_a_setting_error():
    x, y, mask, base = _window()
    with pytest.raises(ValueError, match="noise_model"):
        fit_api.run_fit_consistent(x, y, _seeds(), {"solver": "modern_vp", "noise_model": "poisson"}, base, "add", mask)


def test_per_spectrum_estimates_the_model_and_refits_weighted():
    x, y, mask, base = _window()
    cfg = {"solver": "modern_vp", "noise_model": "per_spectrum"}
    res = fit_api.run_fit_consistent(x, y, _seeds(), cfg, base, "add", mask)
    assert res["fit_ok"]
    nm = res["noise_model"]
    assert set(nm) == {"s0", "g", "n", "method"}
    assert nm["method"] == "fit_residuals_d4"
    assert nm["n"] == int(mask.sum())
    # the measured detector: s0^2 = 0.724 + 0.0451 * 2.4 = 0.83 counts^2, g = 0.0451
    assert 0.6 < nm["s0"] < 1.2
    assert 0.0 < nm["g"] < 0.2
    sigma = res["noise_sigma"]
    assert sigma.shape == (int(mask.sum()),)
    assert sigma.min() == pytest.approx(nm["s0"], rel=0.01)  # far from the lines S ~ 0
    xf, yf = x[mask], y[mask]
    grad_w = _gradient(xf, yf, BACKGROUND, res["theta"], sigma)
    grad_u = _gradient(xf, yf, BACKGROUND, res["theta"], np.ones_like(sigma))
    assert np.max(np.abs(grad_w)) < 1e-4
    assert np.max(np.abs(grad_u)) > 1e-3
    # S_i is the fitted signal above the baseline, taken from the first fit, so
    # sigma_i follows the model at the final fit to well within a percent.
    S = _signal(xf, res["theta"])
    np.testing.assert_allclose(sigma, np.sqrt(nm["s0"] ** 2 + nm["g"] * S), rtol=1e-2)


@pytest.mark.parametrize("solver", ["modern_vp", "modern_trf", "classic"])
def test_a_fixed_sigma_weights_every_solver(solver):
    x, y, mask, base = _window(22)
    xf, yf = x[mask], y[mask]
    truth = [v for line in ((382.92, 68.7, 2.32, 0.5), (408.25, 92.3, 2.75, 0.5)) for v in (line[0], line[1], line[2], line[3])]
    sigma_win = _true_sigma(xf, truth)
    sigma_full = np.ones_like(x)
    sigma_full[mask] = sigma_win
    cfg = {"solver": solver, solver: {"maxfev": 20000}, "noise_sigma": sigma_full}
    res = fit_api.run_fit_consistent(x, y, _seeds(), cfg, base, "add", mask)
    assert res["noise_model"] is None
    np.testing.assert_allclose(res["noise_sigma"], sigma_win)
    assert np.max(np.abs(_gradient(xf, yf, BACKGROUND, res["theta"], sigma_win))) < 1e-4
    plain = fit_api.run_fit_consistent(x, y, _seeds(), {"solver": solver, solver: {"maxfev": 20000}}, base, "add", mask)
    assert np.max(np.abs(_gradient(xf, yf, BACKGROUND, plain["theta"], sigma_win))) > 1e-3


def test_constant_sigma_is_the_unweighted_fit():
    x, y, mask, base = _window(23)
    cfg = {"solver": "modern_vp"}
    plain = fit_api.run_fit_consistent(x, y, _seeds(), cfg, base, "add", mask)
    const = fit_api.run_fit_consistent(
        x, y, _seeds(), {**cfg, "noise_sigma": np.full_like(x, 0.9)}, base, "add", mask
    )
    np.testing.assert_allclose(const["noise_sigma"], 0.9)
    np.testing.assert_allclose(const["theta"], plain["theta"], rtol=1e-7, atol=1e-9)


def test_subtract_mode_uses_the_same_signal():
    x, y, mask, base = _window(24)
    add = fit_api.run_fit_consistent(x, y, _seeds(), {"solver": "modern_vp", "noise_model": "per_spectrum"}, base, "add", mask)
    sub = fit_api.run_fit_consistent(x, y, _seeds(), {"solver": "modern_vp", "noise_model": "per_spectrum"}, base, "subtract", mask)
    assert sub["noise_model"]["s0"] == pytest.approx(add["noise_model"]["s0"], rel=1e-6)
    assert sub["noise_model"]["g"] == pytest.approx(add["noise_model"]["g"], rel=1e-5)
    np.testing.assert_allclose(sub["theta"], add["theta"], rtol=1e-6, atol=1e-8)
