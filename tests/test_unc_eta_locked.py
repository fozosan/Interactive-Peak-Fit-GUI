"""No solver fits eta, so no uncertainty method may treat it as free.

Every solver packs only height, centre and width. The fit's lock mask said eta
was free all the same, the batch asymptotic ignored the lock mask altogether,
and the Bayesian sampled eta. A free eta inflated the FWHM standard error 1.27
to 1.29 times, counted one parameter too many in the degrees of freedom, and
gave eta an interval for a value that never moved. Now the fit marks eta
locked and every method honours the lock mask: dof = n - 3N, and eta has no
standard error.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import fit_api
from core.peaks import Peak
from core.uncertainty_router import route_uncertainty
from tests.synthetic_raman import BACKGROUND, TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

CFG = {"solver": "modern_vp", "modern_vp": {"maxfev": 20000}}


def _fit():
    x, y = mos2_spectrum(11)
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    baseline = np.full_like(x, BACKGROUND)
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    res = fit_api.run_fit_consistent(
        x, y, seeds, CFG, baseline, "add", mask, return_jacobian=True, return_predictors=True
    )
    assert res["fit_ok"]
    return x[mask], y[mask], BACKGROUND, res


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


def test_fit_marks_eta_locked_and_counts_dof_without_it():
    x_fit, _y, _b, res = _fit()
    assert res["locked_mask"].tolist() == [False, False, False, True] * 2
    assert res["dof"] == x_fit.size - 6


def test_asymptotic_se_matches_a_hand_built_jacobian_without_eta():
    x_fit, y_fit, base, res = _fit()
    theta = np.asarray(res["theta"], float)
    unc = route_uncertainty(
        "asymptotic",
        theta_hat=theta,
        residual_fn=res["residual_fn"],
        jacobian=res["jacobian"],
        model_eval=res["predict_fit"],
        fit_ctx={"locked_mask": res["locked_mask"], "bounds": res["bounds"], "x_all": x_fit},
        x_all=x_fit,
        y_all=y_fit,
    )

    J = _hand_jacobian(x_fit, theta)
    r = y_fit - base - _model(x_fit, theta)
    dof = x_fit.size - 6
    cov = (r @ r / dof) * np.linalg.inv(J.T @ J)
    se_hand = np.sqrt(np.diag(cov))
    se_app = [unc.stats[f"p{4 * i + j}"]["sd"] for i in range(2) for j in range(3)]
    np.testing.assert_allclose(se_app, se_hand, rtol=2e-3)
    for i in range(2):
        eta = unc.stats[f"p{4 * i + 3}"]
        assert np.isnan(eta["sd"]) and np.isnan(eta["p2.5"]) and np.isnan(eta["p97.5"])
        assert eta["est"] == 0.5
    assert unc.diagnostics["dof"] == dof


def _model(x, theta):
    out = np.zeros_like(x)
    for c, h, w, e in np.asarray(theta, float).reshape(-1, 4):
        u = (x - c) / w
        out += h * (e / (1.0 + 4.0 * u * u) + (1.0 - e) * np.exp(-4.0 * np.log(2.0) * u * u))
    return out


def test_bayesian_does_not_sample_a_locked_eta():
    x_fit, y_fit, _b, res = _fit()
    theta = np.asarray(res["theta"], float)
    unc = route_uncertainty(
        "bayesian",
        theta_hat=theta,
        residual_fn=res["residual_fn"],
        jacobian=res["jacobian"],
        model_eval=res["predict_fit"],
        fit_ctx={
            "locked_mask": res["locked_mask"],
            "bounds": res["bounds"],
            "bayes_walkers": 0,
            "bayes_burn": 20,
            "bayes_steps": 60,
            "bayes_thin": 1,
        },
        x_all=x_fit,
        y_all=y_fit,
        seed=5,
    )
    for i in range(2):
        eta = unc.stats[f"p{4 * i + 3}"]
        assert eta["est"] == 0.5
        assert np.isnan(eta["sd"]) and np.isnan(eta["p2.5"]) and np.isnan(eta["p97.5"])
        assert np.isfinite(unc.stats[f"p{4 * i + 2}"]["sd"])
    assert np.isnan(unc.stats["eta"]["sd"]).all()


def test_batch_asymptotic_writes_no_eta_error_and_dof_n_minus_3n(tmp_path):
    spectrum = tmp_path / "spec.csv"
    write_spectrum(spectrum, *mos2_spectrum(12))
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(tmp_path / "out"),
        "output_base": "batch",
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        # the peaks' dof alone; a joint baseline (the batch default) also counts
        # its coefficients (tests/test_batch_joint_baseline.py)
        "baseline_fit": "fixed",
    }
    runner.run_batch([str(spectrum)], cfg, compute_uncertainty=True, unc_method="asymptotic")
    with open(tmp_path / "out" / "batch_files.csv", newline="", encoding="utf-8") as fh:
        n_window = int(next(csv.DictReader(fh))["n_points_window"])
    with open(tmp_path / "out" / "spec_uncertainty.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        assert int(row["dof"]) == n_window - 6
        if row["param"] == "eta":
            assert float(row["value"]) == 0.5
            assert row["stderr"] == row["ci_lo"] == row["ci_hi"] == ""
        else:
            assert float(row["stderr"]) > 0
