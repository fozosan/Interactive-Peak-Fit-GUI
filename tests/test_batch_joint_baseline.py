"""The batch fits the baseline with the peaks by default and records it.

Decision D2 (2026-11-08): the GUI holds its baseline fixed; the batch frees it.
``baseline_fit`` is ``"joint"`` in the batch unless the config says ``"fixed"``,
and ``baseline_order`` (default 3) sets the Legendre order over the fit window.
The status table, the fit CSV, the uncertainty CSVs and the batch-wide CSV get
``baseline_fit`` and ``baseline_order``; the TXT gives the coefficients with
their uncertainty. The coefficients never appear as an extra peak.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import baseline_model as BM
from core import fit_api
from core.models import pseudo_voigt
from core.peaks import Peak
from tests.conftest import bayes_knobs
from tests.synthetic_raman import TEMPLATE, WINDOW, curved_shape, mos2_spectrum, write_spectrum

ALS = {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0}
BASELINE_COLUMNS = ("baseline_fit", "baseline_order")


def _config(out_dir, **extra):
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": dict(ALS),
        "output_dir": str(out_dir),
        "output_base": "batch",
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "perf_seed_all": True,
        "perf_seed": 7,
        "perf_unc_workers": 1,
    }
    cfg.update(extra)
    return cfg


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _spectrum(tmp_path, seed=51):
    path = tmp_path / "spec.csv"
    write_spectrum(path, *mos2_spectrum(seed, background=lambda x: 504.9 + curved_shape(x)))
    return path


def _window(path):
    data = np.loadtxt(path, delimiter=",")
    x, y = data[:, 0], data[:, 1]
    m = (x >= WINDOW[0]) & (x <= WINDOW[1])
    return x, y, m


def test_batch_fits_the_baseline_jointly_by_default_and_records_it(tmp_path):
    spec = _spectrum(tmp_path)
    out = tmp_path / "out"
    runner.run_batch([str(spec)], _config(out), compute_uncertainty=True, unc_method="asymptotic")
    header, (status,) = _rows(out / "batch_files.csv")
    assert status["status"] == "ok"
    assert (status["baseline_fit"], status["baseline_order"]) == ("joint", "3")
    fit_header, fit_rows = _rows(out / "spec_fit.csv")
    assert all(c in fit_header for c in BASELINE_COLUMNS)
    assert [(r["baseline_fit"], r["baseline_order"]) for r in fit_rows] == [("joint", "3")] * 2
    unc_header, unc_rows = _rows(out / "spec_uncertainty.csv")
    assert all(c in unc_header for c in BASELINE_COLUMNS)
    # two peaks, four parameters each: the coefficients are not a third peak
    assert sorted({r["peak"] for r in unc_rows}) == ["1", "2"] and len(unc_rows) == 8
    assert {(r["baseline_fit"], r["baseline_order"]) for r in unc_rows} == {("joint", "3")}
    _x, _y, m = _window(spec)
    assert {r["dof"] for r in unc_rows} == {str(int(m.sum()) - 6 - 4)}
    long_header, long_rows = _rows(out / "batch_uncertainty_long.csv")
    assert all(c in long_header for c in BASELINE_COLUMNS)
    assert {(r["baseline_fit"], r["baseline_order"]) for r in long_rows} == {("joint", "3")}
    txt = (out / "spec_uncertainty.txt").read_text(encoding="utf-8")
    assert "Baseline: joint, Legendre order 3" in txt
    for j in range(4):
        line = next(ln for ln in txt.splitlines() if ln.strip().startswith(f"c{j} "))
        assert "±" in line and "95% CI" in line


def test_fixed_holds_the_als_baseline_as_before(tmp_path):
    spec = _spectrum(tmp_path)
    out = tmp_path / "out"
    cfg = _config(out, baseline_fit="fixed")
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="asymptotic")
    _h, (status,) = _rows(out / "batch_files.csv")
    assert (status["baseline_fit"], status["baseline_order"]) == ("fixed", "")
    _h, fit_rows = _rows(out / "spec_fit.csv")
    theta_batch = np.array([float(r[k]) for r in fit_rows for k in ("center", "height", "fwhm", "eta")])
    # the core fit with the batch's own ALS baseline on the window
    x, y, m = _window(spec)
    from core import signals

    z = signals.als_baseline(y[m], lam=1e5, p=0.001, niter=10, tol=0.0)
    base = np.interp(x, x[m], z, left=z[0], right=z[-1])
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    core_cfg = {k: v for k, v in cfg.items() if k != "peaks"}
    core_cfg["noise_model"] = "per_spectrum"
    res = fit_api.run_fit_consistent(
        x, y, seeds, core_cfg, base, "add", m, rng_seed=runner._file_seed(spec, 7)
    )
    np.testing.assert_allclose(theta_batch, res["theta"], rtol=1e-9, atol=1e-9)
    txt = (out / "spec_uncertainty.txt").read_text(encoding="utf-8")
    assert "Baseline: fixed (als)" in txt


def test_an_unknown_baseline_setting_stops_the_batch_before_any_file(tmp_path):
    spec = _spectrum(tmp_path)
    for bad in ({"baseline_fit": "spline"}, {"baseline_order": -2}):
        with pytest.raises(ValueError, match="baseline_"):
            runner.run_batch([str(spec)], _config(tmp_path / "out", **bad), compute_uncertainty=False)
        assert not (tmp_path / "out" / "spec_fit.csv").exists()


def test_batch_and_core_agree_on_the_joint_fit(tmp_path):
    spec = _spectrum(tmp_path, seed=52)
    out = tmp_path / "out"
    cfg = _config(out)
    runner.run_batch([str(spec)], cfg, compute_uncertainty=False)
    _h, fit_rows = _rows(out / "spec_fit.csv")
    theta_batch = np.array([float(r[k]) for r in fit_rows for k in ("center", "height", "fwhm", "eta")])
    x, y, m = _window(spec)
    seeds = [Peak(p["center"], p["height"], p["fwhm"], p["eta"]) for p in TEMPLATE]
    core_cfg = {k: v for k, v in cfg.items() if k != "peaks"}
    core_cfg.update(noise_model="per_spectrum", baseline_fit="joint")
    res = fit_api.run_fit_consistent(x, y, seeds, core_cfg, None, "add", m, rng_seed=runner._file_seed(spec, 7))
    assert res["baseline_fit"] == "joint" and len(res["baseline_coef"]) == 4
    assert [r["baseline_fit"] for r in fit_rows] == ["joint", "joint"]
    np.testing.assert_allclose(theta_batch, res["theta"], rtol=1e-9, atol=1e-9)
    assert float(fit_rows[0]["rmse"]) == pytest.approx(res["rmse"], rel=1e-9)


def test_asymptotic_se_is_the_joint_covariance(tmp_path):
    # Unweighted (noise_model none), so the hand-built covariance is exact:
    # (J^T J)^-1 RSS / (n - 6 - 4), J with the peak columns and the cubic's.
    spec = _spectrum(tmp_path, seed=53)
    out = tmp_path / "out"
    runner.run_batch([str(spec)], _config(out, noise_model="none"), compute_uncertainty=True, unc_method="asymptotic")
    _h, fit_rows = _rows(out / "spec_fit.csv")
    theta = np.array([float(r[k]) for r in fit_rows for k in ("center", "height", "fwhm", "eta")])
    x, y, m = _window(spec)
    xf, yf = x[m], y[m]
    B = BM.JointBaseline.over(xf, 3).basis(xf)

    def peaks(th):
        return sum(pseudo_voigt(xf, h, c, w, e) for c, h, w, e in th.reshape(-1, 4))

    coef = np.linalg.lstsq(B, yf - peaks(theta), rcond=None)[0]
    r = peaks(theta) + B @ coef - yf
    cols = []
    for i in (0, 1, 2, 4, 5, 6):
        th = theta.copy()
        step = 1e-6 * max(1.0, abs(theta[i]))
        th[i] += step
        cols.append((peaks(th) - peaks(theta)) / step)
    J = np.column_stack(cols + list(B.T))
    se = np.sqrt(np.diag(np.linalg.inv(J.T @ J)) * float(r @ r) / (xf.size - 10))
    _h, rows = _rows(out / "spec_uncertainty.csv")
    got = {(int(r_["peak"]), r_["param"]): float(r_["stderr"]) for r_ in rows if r_["stderr"]}
    want = dict(zip([(1, "center"), (1, "height"), (1, "fwhm"), (2, "center"), (2, "height"), (2, "fwhm")], se[:6]))
    for key, val in want.items():
        assert got[key] == pytest.approx(val, rel=3e-3), key
    txt = (out / "spec_uncertainty.txt").read_text(encoding="utf-8")
    c0 = next(ln for ln in txt.splitlines() if ln.strip().startswith("c0 "))
    assert float(c0.split("±")[1].split()[0]) == pytest.approx(se[6], rel=1e-2)


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap", "bayesian"])
def test_every_method_carries_the_coefficients(tmp_path, method, monkeypatch):
    spec = _spectrum(tmp_path, seed=54)
    out = tmp_path / "out"
    cfg = _config(out, bootstrap_n=20, **bayes_knobs(walkers=0, burn=100, steps=300, thin=1))
    refits = []
    real = fit_api.run_fit_consistent

    def spy(x, y, peaks_in, cfg_, *a, **k):
        res = real(x, y, peaks_in, cfg_, *a, **k)
        refits.append((cfg_.get("baseline_fit"), None if res.get("baseline_coef") is None else len(res["baseline_coef"])))
        return res

    monkeypatch.setattr(fit_api, "run_fit_consistent", spy)
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method=method)
    _h, (status,) = _rows(out / "batch_files.csv")
    assert status["unc_status"] == "ok", status["unc_message"]
    _h, rows = _rows(out / "spec_uncertainty.csv")
    assert sorted({r["peak"] for r in rows}) == ["1", "2"]
    txt = (out / "spec_uncertainty.txt").read_text(encoding="utf-8")
    for j in range(4):
        line = next(ln for ln in txt.splitlines() if ln.strip().startswith(f"c{j} "))
        sd = float(line.split("±")[1].split()[0])
        assert np.isfinite(sd) and sd > 0.0
    # the file's fit and, for the bootstrap, every refit fit the baseline jointly
    assert all(r == ("joint", 4) for r in refits)
    assert len(refits) >= 21 if method == "bootstrap" else len(refits) == 1
