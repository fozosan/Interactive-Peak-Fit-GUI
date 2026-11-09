"""The uncertainty of a joint fit includes the baseline (game plan 1.6, decision D2).

With the baseline fitted, its coefficients are parameters of the model like the
peak heights, so every method carries them:

- asymptotic: the Jacobian gains the baseline columns, so the peak covariance is
  the block of (J^T W J)^-1 that includes the baseline, and dof = n - 3N - m;
- bootstrap: each refit fits the baseline again; peaks are relabelled by centre,
  the baseline coefficients stay where they are;
- Bayesian: the coefficients are linear, so the likelihood is integrated over
  them exactly (flat prior): the residual is projected off the baseline columns
  and the scale counts n - m points. Their own posterior comes from the exact
  conditional, given each draw.
"""

import numpy as np
import pytest

from core import baseline_model as BM
from core import fit_api
from core import uncertainty as U
from core.models import pseudo_voigt
from core.peaks import Peak
from core.uncertainty_router import route_uncertainty
from tests.conftest import bayes_knobs


def _signal(x, theta):
    return sum(pseudo_voigt(x, h, c, w, e) for c, h, w, e in np.asarray(theta, float).reshape(-1, 4))


def _broad_problem(seed=3, order=1):
    """A weak broad line in a short window: the baseline and the height trade off."""
    x = np.linspace(490.0, 550.0, 83)
    truth = np.array([520.0, 12.0, 9.0, 0.5])
    coef = np.array([530.0, 1.5, 0.0, 0.0])[: order + 1]
    jb = BM.JointBaseline.over(x, order)
    rng = np.random.default_rng(seed)
    y = _signal(x, truth) + jb.basis(x) @ coef + 1.0 * rng.standard_normal(x.size)
    return x, y, truth, jb


def _joint_fit(x, y, order=1):
    cfg = {"solver": "modern_vp", "baseline_fit": "joint", "baseline_order": order}
    return fit_api.run_fit_consistent(
        x, y, [Peak(519.0, 10.0, 8.0, 0.5)], cfg, None, "add", np.ones_like(x, bool), return_jacobian=True
    )


def _problem(res, x, y):
    return BM.uncertainty_problem(res, x, y)


def test_the_problem_carries_the_coefficients_as_parameters():
    x, y, _truth, jb = _broad_problem()
    res = _joint_fit(x, y)
    pb = _problem(res, x, y)
    m = jb.n_coef
    np.testing.assert_allclose(pb["theta"], np.r_[res["theta"], res["baseline_coef"]])
    assert pb["param_names"][-m:] == ["baseline_c0", "baseline_c1"]
    assert pb["n_peak_params"] == 4
    np.testing.assert_array_equal(pb["locked_mask"], [False, False, False, True, False, False])
    assert np.all(np.isinf(pb["bounds"][0][-m:])) and np.all(np.isinf(pb["bounds"][1][-m:]))
    # the predictor is the model on the window, the residual model - data
    np.testing.assert_allclose(pb["predict"](pb["theta"]), y + res["residual"], atol=1e-9)
    np.testing.assert_allclose(pb["residual_fn"](pb["theta"]), res["residual"], atol=1e-9)
    J = pb["jacobian"]
    assert J.shape == (x.size, 6)
    np.testing.assert_allclose(J[:, 4:], jb.basis(x))


def test_asymptotic_peak_covariance_includes_the_baseline_block():
    x, y, _truth, jb = _broad_problem()
    res = _joint_fit(x, y)
    pb = _problem(res, x, y)
    out = route_uncertainty(
        "asymptotic", theta_hat=pb["theta"], residual_fn=pb["residual_fn"], jacobian=pb["jacobian"],
        model_eval=pb["predict"], fit_ctx={"locked_mask": pb["locked_mask"], "bounds": pb["bounds"], "x_all": x},
        x_all=x, y_all=y,
    )
    # hand-built: free columns centre, height, FWHM and the two coefficients
    J = pb["jacobian"][:, [0, 1, 2, 4, 5]]
    r = pb["residual_fn"](pb["theta"])
    dof = x.size - 5
    cov = np.linalg.inv(J.T @ J) * float(r @ r) / dof
    se = np.sqrt(np.diag(cov))
    got = np.array([out.stats[f"p{i}"]["sd"] for i in (0, 1, 2, 4, 5)])
    np.testing.assert_allclose(got, se, rtol=2e-3)
    assert out.diagnostics["dof"] == dof
    # without the baseline's columns (a fixed baseline) the FWHM interval is
    # about 15 % narrower here: the line's wings trade off against the baseline
    se_fixed = np.sqrt(np.diag(np.linalg.inv(J[:, :3].T @ J[:, :3]) * float(r @ r) / (x.size - 3)))
    assert se[2] > 1.1 * se_fixed[2]


def test_bootstrap_relabels_the_peaks_and_leaves_the_coefficients_in_place():
    # The coefficient c0 (383.5 counts) sits nearer the first centre than the
    # moved peak does; relabelling by centre must not take it for a peak.
    x = np.linspace(340.0, 440.0, 60)
    theta = np.array([383.0, 60.0, 2.3, 0.5, 408.0, 80.0, 2.8, 0.5, 383.5, 1.0, 0.2, 0.1])
    jb = BM.JointBaseline.over(x, 3)

    def predict(th):
        return _signal(x, th[:8]) + jb.basis(x) @ th[8:]

    y = predict(theta)
    peaks = [Peak(383.0, 60.0, 2.3, 0.5), Peak(408.0, 80.0, 2.8, 0.5)]

    def refit(theta_init, locked, bounds, xx, yy):
        t = np.asarray(theta_init, float)
        out = np.r_[t[4:8], t[0:4], t[8:]]  # the peaks come back swapped
        out[4] += 0.9  # the first peak moved 0.9 cm-1
        out[8] -= 0.1  # c0 moved 0.1 count
        return out, True

    locked = np.array([False, False, False, True] * 2 + [False] * 4)
    res = U.bootstrap_ci(
        theta=theta, residual=np.zeros(x.size), jacobian=np.zeros((x.size, 12)), predict_full=predict,
        x_all=x, y_all=y, fit_ctx={"refit": refit, "peaks": peaks, "solver": "modern_vp", "strict_refit": True},
        param_names=[f"p{i}" for i in range(12)], locked_mask=locked, n_boot=4, seed=1,
    )
    st = res.stats
    assert st["p0"]["sd"] == pytest.approx(0.9)  # peak 1's centre moved
    assert st["p4"]["sd"] == pytest.approx(0.0)  # peak 2 kept its identity
    assert st["p8"]["sd"] == pytest.approx(0.1)  # c0 stayed a coefficient
    assert st["p9"]["sd"] == pytest.approx(0.0)


def test_bayesian_integrates_over_the_baseline():
    x, y, _truth, jb = _broad_problem(seed=4)
    res = _joint_fit(x, y)
    pb = _problem(res, x, y)
    asy = route_uncertainty(
        "asymptotic", theta_hat=pb["theta"], residual_fn=pb["residual_fn"], jacobian=pb["jacobian"],
        model_eval=pb["predict"], fit_ctx={"locked_mask": pb["locked_mask"], "bounds": pb["bounds"], "x_all": x},
        x_all=x, y_all=y,
    )
    # A finite height ceiling: the solver's 1e12 sentinel starts the walkers
    # +-1e9 counts out (review finding 6, PR F), which is not under test here.
    lo, hi = (np.array(b, float) for b in pb["bounds"])
    hi[1] = 100.0
    ctx = {
        "locked_mask": pb["locked_mask"], "bounds": (lo, hi), "x_all": x, "y_all": y,
        "param_names": pb["param_names"], "baseline_basis": jb.basis(x),
        **bayes_knobs(walkers=0, burn=400, steps=2500, thin=1),
    }
    np.random.seed(2)  # emcee draws its moves from numpy's global state (review finding 6)
    bay = route_uncertainty(
        "bayesian", theta_hat=pb["theta"], residual_fn=pb["residual_fn"], jacobian=pb["jacobian"],
        model_eval=pb["predict"], fit_ctx=ctx, x_all=x, y_all=y, seed=11,
    )
    st = bay.stats
    # one peak only: the coefficients are not a phantom second peak
    assert len(st["center"]["est"]) == 1
    sd_bay = np.array([st["center"]["sd"][0], st["height"]["sd"][0], st["fwhm"]["sd"][0]])
    sd_asy = np.array([asy.stats[f"p{i}"]["sd"] for i in (0, 1, 2)])
    np.testing.assert_allclose(sd_bay, sd_asy, rtol=0.2)
    for j in range(jb.n_coef):
        blk = st[f"baseline_c{j}"]
        assert blk["sd"] == pytest.approx(asy.stats[f"p{4 + j}"]["sd"], rel=0.2)
        assert blk["est"] == pytest.approx(pb["theta"][4 + j], abs=2.0 * blk["sd"])
    assert bay.diagnostics.get("baseline_marginalized") == jb.n_coef
