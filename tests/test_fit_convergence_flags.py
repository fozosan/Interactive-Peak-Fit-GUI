"""A fit reports whether it converged, and flags a parameter on its bound.

Modern VP returned ok=True on every exit, also after running out of
evaluations, so the batch status said "ok" and the bootstrap pooled refits
that had stopped anywhere. Hitting a bound is a flag, never a veto: the fit is
kept and the status table's ``flags`` column says ``bound_hit``.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core import fit_api
from core.models import pseudo_voigt
from core.peaks import Peak
from fit import modern_vp


def _spectrum(spike_fwhm=None):
    rng = np.random.default_rng(4)
    x = np.linspace(340.0, 440.0, 135)
    y = 505.0 + pseudo_voigt(x, 60.0, 383.0, 2.4, 0.5) + rng.normal(0.0, 0.9, x.size)
    if spike_fwhm is not None:  # narrower than the 2-point FWHM floor
        y = y + pseudo_voigt(x, 30.0, 408.0, spike_fwhm, 0.5)
    return x, y


def test_modern_vp_reports_running_out_of_evaluations():
    x, y = _spectrum()
    base = np.full_like(x, 505.0)
    stopped = modern_vp.solve(x, y, [Peak(370.0, 50.0, 6.0, 0.5)], "add", base, {"maxfev": 1})
    assert stopped["ok"] is False
    assert "maxfev" in stopped["message"]
    done = modern_vp.solve(x, y, [Peak(383.5, 50.0, 3.0, 0.5)], "add", base, {"maxfev": 200})
    assert done["ok"] is True


def test_a_fit_resting_on_its_bound_converges_and_says_so():
    # A line 8 wide under a width cap of 5: the optimum is on the cap. The
    # Gauss-Newton step keeps pointing past it, so a test on the step size
    # never passed and the solver ran all its evaluations; the move it made
    # is what shrinks to zero.
    x = np.linspace(20.0, 80.0, 121)
    y = 10.0 + pseudo_voigt(x, 100.0, 50.0, 8.0, 0.5)
    seed = [Peak(49.0, 80.0, 4.0, 0.5)]
    opts = {"maxfev": 2000, "width_caps": [5.0], "centers_in_window": True}
    res = modern_vp.solve(x, y, seed, "add", np.full_like(x, 10.0), opts)
    assert res["ok"] is True
    assert res["meta"]["nfev"] < 200
    assert res["theta"][2] == pytest.approx(5.0, abs=1e-6)

    fit = fit_api.run_fit_consistent(
        x, y, seed, {"solver": "modern_vp", "modern_vp": opts}, np.full_like(x, 10.0), "add",
        np.ones_like(x, bool),
    )
    assert fit["fit_ok"] is True and fit["bound_hit"] == ["fwhm1"]


def test_run_fit_consistent_carries_convergence_and_bound_flags():
    x, y = _spectrum(spike_fwhm=0.3)
    base = np.full_like(x, 505.0)
    seeds = [Peak(383.0, 50.0, 3.0, 0.5), Peak(408.0, 20.0, 3.0, 0.5)]
    mask = np.ones_like(x, bool)
    res = fit_api.run_fit_consistent(x, y, seeds, {"solver": "modern_vp"}, base, "add", mask)
    assert res["fit_ok"] is True
    assert res["flags"] == ["bound_hit"]
    assert res["bound_hit"] == ["fwhm2"]

    clean = fit_api.run_fit_consistent(x, y, seeds[:1], {"solver": "modern_vp"}, base, "add", mask)
    assert clean["flags"] == [] and clean["bound_hit"] == []

    cut = fit_api.run_fit_consistent(
        x, y, [Peak(370.0, 50.0, 6.0, 0.5)], {"solver": "modern_vp", "solver_maxfev": 1}, base, "add", mask
    )
    assert cut["fit_ok"] is False


def _batch(tmp_path, name, spike_fwhm, **extra):
    x, y = _spectrum(spike_fwhm)
    path = tmp_path / f"{name}.csv"
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")
    cfg = {
        "peaks": [
            {"center": 383.0, "height": 50.0, "fwhm": 3.0, "eta": 0.5},
            {"center": 408.0, "height": 20.0, "fwhm": 3.0, "eta": 0.5},
        ],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "polynomial", "degree": 0},
        "output_dir": str(tmp_path / name),
        "output_base": "batch",
        **extra,
    }
    runner.run_batch([str(path)], cfg, compute_uncertainty=False)
    with open(tmp_path / name / "batch_files.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, next(reader)


def test_status_table_flags_a_bound_and_reports_non_convergence(tmp_path):
    header, row = _batch(tmp_path, "spike", spike_fwhm=0.3)
    assert header[header.index("status") + 1] == "flags"
    assert row["status"] == "ok" and row["flags"] == "bound_hit"
    assert "fwhm2" in row["message"]

    _, row = _batch(tmp_path, "cut", spike_fwhm=None, solver_maxfev=1)
    assert row["status"] == "not_converged"
    assert row["fit_ok"] == "False"
