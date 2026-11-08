"""Every uncertainty method uses the same per-point sigma_i as the weighted fit.

- Asymptotic: the weighted Jacobian covariance, (J_w^T J_w)^-1 x RSS_w / dof, with
  J_w = J / sigma and r_w = r / sigma.
- Bootstrap: the residuals are standardised, e_i = r_i / sigma_i, centred, resampled
  i.i.d. and rescaled by the sigma_i of the point they land on.
- Bayesian: a Gaussian likelihood with sd k * sigma_i at each point, k sampled as
  the old single sigma was.

With a constant sigma each method reduces exactly to the unweighted one.
"""

import numpy as np
import pytest

from core import fit_api
from core import uncertainty as U
from core.models import pseudo_voigt
from core.peaks import Peak
from core.uncertainty_router import route_uncertainty
from tests.synthetic_raman import (
    BACKGROUND, BIAS, MOS2_LINES, NOISE_FLOOR, NOISE_GAIN, TEMPLATE, WINDOW, mos2_spectrum,
)

TRUTH = np.array([v for line in MOS2_LINES for v in line], float)


def _model(x, theta):
    return sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in np.asarray(theta, float).reshape(-1, 4))


def _hand_jacobian(x, theta):
    """d(model)/d(centre, height, fwhm) of each pseudo-Voigt, analytically: no eta columns."""
    cols = []
    for c, h, w, e in np.asarray(theta, float).reshape(-1, 4):
        u = (x - c) / w
        lor = 1.0 / (1.0 + 4.0 * u * u)
        gau = np.exp(-4.0 * np.log(2.0) * u * u)
        dprof_du = e * (-8.0 * u) * lor * lor + (1.0 - e) * gau * (-8.0 * np.log(2.0) * u)
        cols += [h * dprof_du * (-1.0 / w), e * lor + (1.0 - e) * gau, h * dprof_du * (-u / w)]
    return np.column_stack(cols)


def _weighted_fit(seed=31):
    x, y = mos2_spectrum(seed)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    xf = x[mask]
    sigma_win = np.sqrt(NOISE_FLOOR + NOISE_GAIN * (BACKGROUND - BIAS + _model(xf, TRUTH)))
    sigma_full = np.ones_like(x)
    sigma_full[mask] = sigma_win
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    cfg = {"solver": "modern_vp", "modern_vp": {"maxfev": 20000}, "noise_sigma": sigma_full}
    res = fit_api.run_fit_consistent(
        x, y, seeds, cfg, np.full_like(x, BACKGROUND), "add", mask,
        return_jacobian=True, return_predictors=True,
    )
    assert res["fit_ok"]
    return xf, y[mask], res, sigma_win


def _ctx(res, xf, sigma=None):
    ctx = {"locked_mask": res["locked_mask"], "bounds": res["bounds"], "x_all": xf}
    if sigma is not None:
        ctx["noise_sigma"] = sigma
    return ctx


def test_asymptotic_se_is_the_weighted_covariance():
    xf, yf, res, sigma = _weighted_fit()
    theta = np.asarray(res["theta"], float)
    unc = route_uncertainty(
        "asymptotic", theta_hat=theta, residual_fn=res["residual_fn"], jacobian=res["jacobian"],
        model_eval=res["predict_fit"], fit_ctx=_ctx(res, xf, sigma), x_all=xf, y_all=yf,
    )
    Jw = _hand_jacobian(xf, theta) / sigma[:, None]
    rw = (yf - BACKGROUND - _model(xf, theta)) / sigma
    cov = (rw @ rw / (xf.size - 6)) * np.linalg.inv(Jw.T @ Jw)
    se_hand = np.sqrt(np.diag(cov))
    se_app = [unc.stats[f"p{4 * i + j}"]["sd"] for i in range(2) for j in range(3)]
    np.testing.assert_allclose(se_app, se_hand, rtol=2e-3)
    assert unc.diagnostics["noise_weighted"] is True
    # The weights matter: the unweighted formula gives a different answer.
    plain = route_uncertainty(
        "asymptotic", theta_hat=theta, residual_fn=res["residual_fn"], jacobian=res["jacobian"],
        model_eval=res["predict_fit"], fit_ctx=_ctx(res, xf), x_all=xf, y_all=yf,
    )
    assert plain.diagnostics["noise_weighted"] is False
    assert plain.stats["p1"]["sd"] < 0.8 * unc.stats["p1"]["sd"]


def test_asymptotic_with_a_constant_sigma_is_the_unweighted_one():
    xf, yf, res, _ = _weighted_fit(32)
    theta = np.asarray(res["theta"], float)
    kw = dict(theta_hat=theta, residual=res["residual"], jacobian=res["jacobian"],
              ymodel_fn=res["predict_fit"], locked_mask=res["locked_mask"], bounds=res["bounds"], xgrid=xf)
    plain = U.asymptotic_ci(**kw)
    const = U.asymptotic_ci(**kw, sigma=np.full_like(xf, 0.9))
    assert const.diagnostics["noise_weighted"] is True
    for i in range(theta.size):
        a, b = plain.stats[f"p{i}"], const.stats[f"p{i}"]
        np.testing.assert_allclose([b["sd"], b["p2.5"], b["p97.5"]], [a["sd"], a["p2.5"], a["p97.5"]], rtol=1e-10)


def _bootstrap_draws(sigma_kind, seed=5, n_boot=12):
    """The synthetic data each refit receives, with a refit that only records them.

    ``sigma_kind``: None (unweighted), "model" (the measured noise model) or a
    constant.
    """
    xf, yf, res, sig_model = _weighted_fit(33)
    sigma = sig_model if sigma_kind == "model" else (
        None if sigma_kind is None else np.full_like(xf, float(sigma_kind))
    )
    theta = np.asarray(res["theta"], float)
    seen = []

    def refit(theta_init, locked_mask, bounds, x, y):
        seen.append(np.asarray(y, float).copy())
        return np.asarray(theta_init, float), True

    yhat = res["predict_fit"](theta)
    U.bootstrap_ci(
        theta=theta, residual=res["residual"], jacobian=res["jacobian"], predict_full=res["predict_fit"],
        x_all=xf, y_all=yf, fit_ctx={"refit": refit, "strict_refit": True}, n_boot=n_boot, seed=seed,
        locked_mask=res["locked_mask"], bounds=res["bounds"], center_residuals=True, sigma=sigma,
    )
    return xf, yf, yhat, np.array(seen), sig_model


def test_bootstrap_resamples_standardised_residuals_rescaled_by_sigma():
    xf, yf, yhat, draws, sigma = _bootstrap_draws("model")
    e = (yf - yhat) / sigma
    e = e - e.mean()
    for y_b in draws:
        e_b = (y_b - yhat) / sigma
        # every draw is a standardised residual of the fit, rescaled at its new point
        assert np.all(np.min(np.abs(e_b[:, None] - e[None, :]), axis=1) < 1e-9)
    # i.i.d.: residuals move between points (the peak-top sigma is not kept in place)
    assert not np.allclose(draws[0] - yhat, (yf - yhat) - np.mean(yf - yhat))


def test_bootstrap_with_a_constant_sigma_is_the_unweighted_one():
    _x, _y, _yh, draws_plain, _s = _bootstrap_draws(None)
    _x, _y, _yh, draws_const, _s = _bootstrap_draws(0.9)
    np.testing.assert_allclose(draws_const, draws_plain, rtol=1e-12, atol=1e-9)


def _bayes(res, xf, yf, sigma, seed=11):
    """Finite physical bounds, as the GUI builds them: the solver's open height
    bound would start the walkers far wider than the posterior (game plan 1.7)."""
    P = np.asarray(res["theta"]).size
    lo, hi = np.zeros(P), np.ones(P)
    lo[0::4], hi[0::4] = xf.min(), xf.max()
    hi[1::4] = 1.5 * float(np.max(yf - BACKGROUND))
    lo[2::4], hi[2::4] = 0.5, xf.max() - xf.min()
    np.random.seed(0)  # emcee draws its moves from a copy of the global state
    return U.bayesian_ci(
        theta_hat=np.asarray(res["theta"], float), model=res["predict_fit"], predict_full=res["predict_fit"],
        x_all=xf, y_all=yf, residual_fn=res["residual_fn"], bounds=(lo, hi),
        locked_mask=res["locked_mask"], fit_ctx={"x_all": xf}, n_walkers=None, n_burn=400, n_steps=1600,
        thin=1, seed=seed, return_band=False, sigma=sigma,
    )


def test_bayesian_with_a_constant_sigma_is_the_unweighted_posterior():
    """sigma = 1 everywhere is the unweighted sampler, draw for draw. Any other
    constant only rescales the sampled k, so the posterior of the peaks is the
    same; the chains themselves part after a while, because the ensemble
    amplifies rounding differences, so that case is compared within Monte Carlo
    error."""
    xf, yf, res, _ = _weighted_fit(34)
    plain = _bayes(res, xf, yf, None)
    ones = _bayes(res, xf, yf, np.ones_like(xf))
    for i in range(8):
        a, b = plain.stats[f"p{i}"], ones.stats[f"p{i}"]
        np.testing.assert_allclose([b["est"], b["sd"]], [a["est"], a["sd"]], rtol=1e-12, atol=1e-14)
    assert ones.diagnostics["noise_weighted"] is True
    assert plain.diagnostics["noise_weighted"] is False
    half = _bayes(res, xf, yf, np.full_like(xf, 0.5))
    for i in (0, 1, 2, 4, 5, 6):
        a, b = plain.stats[f"p{i}"], half.stats[f"p{i}"]
        assert b["est"] == pytest.approx(a["est"], abs=0.2 * a["sd"])
        assert b["sd"] == pytest.approx(a["sd"], rel=0.15)


def test_bayesian_posterior_follows_the_per_point_sigma():
    """The posterior sd matches the weighted asymptotic SE, not the unweighted one."""
    xf, yf, res, sigma = _weighted_fit(35)
    theta = np.asarray(res["theta"], float)
    post = _bayes(res, xf, yf, sigma)
    asy = route_uncertainty(
        "asymptotic", theta_hat=theta, residual_fn=res["residual_fn"], jacobian=res["jacobian"],
        model_eval=res["predict_fit"], fit_ctx=_ctx(res, xf, sigma), x_all=xf, y_all=yf,
    )
    plain = route_uncertainty(
        "asymptotic", theta_hat=theta, residual_fn=res["residual_fn"], jacobian=res["jacobian"],
        model_eval=res["predict_fit"], fit_ctx=_ctx(res, xf), x_all=xf, y_all=yf,
    )
    for i in (1, 5):  # the two heights, where the weighting matters most
        ratio_w = post.stats[f"p{i}"]["sd"] / asy.stats[f"p{i}"]["sd"]
        ratio_u = post.stats[f"p{i}"]["sd"] / plain.stats[f"p{i}"]["sd"]
        assert 0.75 < ratio_w < 1.3, ratio_w
        assert ratio_u > 1.3, ratio_u
