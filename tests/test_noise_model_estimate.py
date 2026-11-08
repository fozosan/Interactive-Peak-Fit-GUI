"""One noise model per spectrum: var_i = s0^2 + g * S_i (game plan 1.5, decision D1).

S is the fitted signal above the baseline, in counts. The model is estimated from
a fit's residuals r through their 4th differences d = C r, which cancel a smooth
misfit (a baseline that bends into the lines, a fixed eta): s0 is 1.4826 x MAD
(cafm's ``robust_sigma`` convention) of the standardised d, and g >= 0 comes from
a robust regression of d^2 on its exact expected variance, which also carries
the fit's shrinkage of its residuals (leverage). Difference order 0 uses r
itself. With g = 0 it is one constant sigma, the noise every method assumed
before.

The 2026-11-07 review measured var = 0.724 + 0.0451 * S on the user's CCD
spectra (S above the bias), so a 90-count peak top has 6.6 times the floor
variance.
"""

import numpy as np
import pytest

from core.noise_model import (
    NoiseModel, difference_stencil, estimate_noise_model, robust_sigma, variance_regressors,
)


def _line(x, hgt=90.0, w=2.5):
    """A pseudo-Voigt (eta 0.5) and its Jacobian over height, centre and FWHM."""
    u = x / w
    lor = 1.0 / (1.0 + 4.0 * u * u)
    gau = np.exp(-4.0 * np.log(2.0) * u * u)
    prof = 0.5 * lor + 0.5 * gau
    dprof = 0.5 * (-8.0 * u) * lor * lor + 0.5 * gau * (-8.0 * np.log(2.0) * u)
    J = np.column_stack([prof, hgt * dprof * (-1.0 / w), hgt * dprof * (-u / w)])
    return hgt * prof, J


def test_robust_sigma_follows_the_cafm_convention():
    # median 3, |x - 3| = 2, 1, 0, 1, 97: MAD 1
    assert robust_sigma([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(1.4826)
    # non-finite values are left out
    assert robust_sigma([1.0, np.nan, 2.0, 3.0, np.inf, 4.0, 100.0]) == pytest.approx(1.4826)
    # empty input gives 0
    assert robust_sigma([]) == 0.0
    assert robust_sigma([np.nan, np.inf]) == 0.0
    # more than half the values identical: MAD = 0 falls back to the std
    y = np.array([5.0, 5.0, 5.0, 5.0, 9.0])
    assert robust_sigma(y) == pytest.approx(np.std(y))


def test_the_filter_is_the_normalised_4th_difference():
    np.testing.assert_allclose(difference_stencil(4), np.array([1, -4, 6, -4, 1]) / np.sqrt(70.0))
    np.testing.assert_allclose(difference_stencil(0), [1.0])
    # a cubic cancels
    x = np.linspace(0.0, 3.0, 40)
    cubic = 2.0 + x - 0.5 * x**2 + 0.3 * x**3
    c = difference_stencil(4)
    assert np.max(np.abs(np.convolve(cubic, c[::-1], mode="valid"))) < 1e-10


@pytest.mark.parametrize("order", [0, 4])
def test_the_regressors_are_the_exact_expected_variance(order):
    """E[(C r)_i^2] = s0^2 (M 1)_i + g (M S)_i, M = (C (I - H)) o (C (I - H))."""
    x = np.arange(-15.0, 15.0, 0.73)
    S, J = _line(x)
    H = J @ np.linalg.pinv(J)
    c = difference_stencil(order)
    m = x.size - c.size + 1
    C = np.zeros((m, x.size))
    for i in range(m):
        C[i, i : i + c.size] = c
    A = C @ (np.eye(x.size) - H)
    M = A * A
    X = variance_regressors(J, S, order)
    np.testing.assert_allclose(X[:, 0], M @ np.ones(x.size), atol=1e-12)
    np.testing.assert_allclose(X[:, 1], M @ S, atol=1e-10)
    if order == 0:  # the hat diagonal: E[r_i^2] = (1 - h_i) s0^2 for constant noise
        np.testing.assert_allclose(X[:, 0], 1.0 - np.diag(H), atol=1e-12)


@pytest.mark.parametrize("order", [0, 4])
def test_estimate_recovers_a_known_model(order):
    rng = np.random.default_rng(0)
    S = rng.uniform(0.0, 200.0, 20000)
    s0, g = 0.9, 0.045
    r = rng.standard_normal(S.size) * np.sqrt(s0 * s0 + g * S)
    model = estimate_noise_model(r, S, order=order)
    assert isinstance(model, NoiseModel)
    assert model.s0 == pytest.approx(s0, rel=0.04)
    assert model.g == pytest.approx(g, rel=0.1)
    assert model.n == S.size
    assert model.method == f"fit_residuals_d{order}"
    np.testing.assert_allclose(model.sigma(S), np.sqrt(model.s0**2 + model.g * S))
    assert model.as_dict() == {"s0": model.s0, "g": model.g, "n": model.n, "method": model.method}


def test_the_default_is_the_4th_difference():
    rng = np.random.default_rng(6)
    S = np.zeros(300)
    r = rng.standard_normal(300)
    assert estimate_noise_model(r, S).method == "fit_residuals_d4"


def test_constant_noise_is_the_g_zero_case():
    """No signal: g = 0. With order 0, s0 is exactly robust_sigma of the residuals."""
    rng = np.random.default_rng(2)
    r = 0.8 * rng.standard_normal(500)
    plain = estimate_noise_model(r, np.zeros_like(r), order=0)
    assert plain.g == 0.0
    assert plain.s0 == pytest.approx(robust_sigma(r), rel=1e-12)
    np.testing.assert_allclose(plain.sigma(np.linspace(0, 100, 7)), plain.s0)
    filtered = estimate_noise_model(r, np.zeros_like(r))
    assert filtered.g == 0.0
    assert filtered.s0 == pytest.approx(0.8, rel=0.1)


def test_g_is_never_negative():
    rng = np.random.default_rng(3)
    S = rng.uniform(0.0, 100.0, 4000)
    r = rng.standard_normal(S.size) * np.sqrt(4.0 - 0.03 * S)  # variance falls with S
    model = estimate_noise_model(r, S)
    assert model.g == 0.0
    assert model.s0 > 0.0


def test_one_spike_barely_moves_the_estimate():
    """A cosmic ray on the baseline counts as at most 9 times its model variance.

    Window-like points: most on the baseline, a fifth under lines up to 150 counts.
    """
    rng = np.random.default_rng(4)
    S = np.where(rng.random(3000) < 0.8, 0.0, rng.uniform(0.0, 150.0, 3000))
    r = rng.standard_normal(S.size) * np.sqrt(0.8 + 0.05 * S)
    clean = estimate_noise_model(r, S)
    r_spike = r.copy()
    r_spike[int(np.argmin(S[10:])) + 10] += 400.0  # ~450 sigma
    spiked = estimate_noise_model(r_spike, S)
    assert spiked.s0 == pytest.approx(clean.s0, rel=0.02)
    assert spiked.g == pytest.approx(clean.g, rel=0.05)
    # without the cap one point decides the model
    uncapped = estimate_noise_model(r_spike, S, cap=np.inf)
    assert abs(uncapped.g / clean.g - 1.0) > 0.5 or abs(uncapped.s0 / clean.s0 - 1.0) > 0.5


def test_the_shrinkage_of_fitted_residuals_is_carried():
    """Residuals of a fit are shrunk by (1 - h), and h is large at a narrow line's top.

    A pseudo-Voigt 3 to 4 points wide on the 800g axis, linearised: the residuals
    of the least-squares fit, r = (I - H) e. From the residuals themselves
    (order 0), g comes out about half the truth without the Jacobian and at the
    truth with it. The 4th differences hardly see the shrinkage, which lies in
    the smooth, line-shaped part of r; their regressors carry it all the same
    (exact, see above).
    """
    order = 0
    x = np.arange(-20.0, 20.0, 0.73)
    S, J = _line(x)
    s0, g = 0.91, 0.0451
    H = J @ np.linalg.pinv(J)
    rng = np.random.default_rng(5)
    g_fit, g_raw = [], []
    for _ in range(300):
        e = rng.standard_normal(x.size) * np.sqrt(s0 * s0 + g * S)
        r = e - H @ e
        g_fit.append(estimate_noise_model(r, S, jacobian=J, order=order).g)
        g_raw.append(estimate_noise_model(r, S, order=order).g)
    assert np.mean(g_raw) < 0.8 * g
    assert np.mean(g_fit) == pytest.approx(g, rel=0.25)


def test_a_smooth_misfit_is_not_read_as_noise():
    """A baseline that bends 3 counts into the line, as ALS does on these spectra.

    The same noise with and without the bend, through a linearised fit: the plain
    residuals (order 0) read the bend as signal-dependent noise, g up 4-fold; the
    4th differences hardly move.
    """
    x = np.arange(-30.0, 30.0, 0.73)
    S, J = _line(x)
    bend = -3.0 * np.exp(-0.5 * (x / 8.0) ** 2)
    rng = np.random.default_rng(7)
    g = {(order, b): [] for order in (0, 4) for b in (0, 1)}
    for _ in range(200):
        e = rng.standard_normal(x.size) * np.sqrt(0.83 + 0.0451 * S)
        for b in (0, 1):
            y = e + b * bend
            r = y - J @ np.linalg.lstsq(J, y, rcond=None)[0]
            for order in (0, 4):
                g[order, b].append(estimate_noise_model(r, S, jacobian=J, order=order).g)
    ratio4 = np.mean(g[4, 1]) / np.mean(g[4, 0])
    ratio0 = np.mean(g[0, 1]) / np.mean(g[0, 0])
    assert ratio4 < 1.25, ratio4
    assert ratio0 > 2.0, ratio0


def test_too_few_points_give_no_model():
    assert estimate_noise_model([0.1, -0.2], [1.0, 2.0], order=0) is None
    assert estimate_noise_model(np.zeros(6), np.ones(6)) is None  # 2 filtered values
    assert estimate_noise_model([np.nan] * 10, np.ones(10)) is None


def test_zero_noise_gives_a_zero_floor_and_no_slope():
    model = estimate_noise_model(np.zeros(50), np.linspace(0, 100, 50))
    assert model.s0 == 0.0 and model.g == 0.0
    assert not model.usable
