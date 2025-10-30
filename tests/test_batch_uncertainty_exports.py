import logging
import numpy as np
import pandas as pd
import pytest

from batch import runner
from core.peaks import Peak
from ui.app import pseudo_voigt


@pytest.mark.parametrize("method", ["Asymptotic", "Bootstrap", "Bayesian"])
def test_batch_uncertainty_exports(tmp_path, method):
    x = np.linspace(-5, 5, 101)
    pk = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
    p = tmp_path / "a.csv"
    np.savetxt(p, np.column_stack([x, y]), delimiter=",")

    cfg = {
        "peaks": [pk.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        "classic": {},
        "baseline_uses_fit_range": True,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 1,
        "output_dir": str(tmp_path),
        "output_base": "batch",
    }

    ok, total = runner.run_batch([str(p)], cfg, compute_uncertainty=True, unc_method=method)
    assert ok == 1 and total == 1

    unc_csv = tmp_path / "a_uncertainty.csv"
    assert unc_csv.exists()
    df = pd.read_csv(unc_csv)
    assert (df["value"].abs() > 0).any()
    assert (df["stderr"].abs() > 0).any()

    band_csv = tmp_path / "a_uncertainty_band.csv"
    if band_csv.exists():
        bdf = pd.read_csv(band_csv)
        assert len(bdf) > 0
        assert np.isfinite(bdf.values).all()



def test_batch_uncertainty_failure_does_not_abort(tmp_path, monkeypatch, caplog):
    x = np.linspace(-5, 5, 101)
    pk = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)

    fail_path = tmp_path / "fail.csv"
    ok_path = tmp_path / "ok.csv"
    for file_path in (fail_path, ok_path):
        np.savetxt(file_path, np.column_stack([x, y]), delimiter=",")

    cfg = {
        "peaks": [pk.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        "classic": {},
        "baseline_uses_fit_range": True,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 1,
        "output_dir": str(tmp_path),
        "output_base": "batch",
        "bootstrap_n": 8,
    }

    stats_entry = {
        "center": {
            "est": float(pk.center),
            "sd": 0.1,
            "ci_lo": pk.center - 0.1,
            "ci_hi": pk.center + 0.1,
            "p2_5": pk.center - 0.1,
            "p97_5": pk.center + 0.1,
        },
        "height": {
            "est": float(pk.height),
            "sd": 0.1,
            "ci_lo": pk.height - 0.1,
            "ci_hi": pk.height + 0.1,
            "p2_5": pk.height - 0.1,
            "p97_5": pk.height + 0.1,
        },
        "fwhm": {
            "est": float(pk.fwhm),
            "sd": 0.1,
            "ci_lo": pk.fwhm - 0.1,
            "ci_hi": pk.fwhm + 0.1,
            "p2_5": pk.fwhm - 0.1,
            "p97_5": pk.fwhm + 0.1,
        },
        "eta": {
            "est": float(pk.eta),
            "sd": 0.05,
            "ci_lo": max(0.0, pk.eta - 0.1),
            "ci_hi": min(1.0, pk.eta + 0.1),
            "p2_5": max(0.0, pk.eta - 0.1),
            "p97_5": min(1.0, pk.eta + 0.1),
        },
    }
    band = (x.copy(), y - 0.1, y + 0.1)

    call_count = {"value": 0}

    def fake_bootstrap_ci(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("synthetic failure")
        return {
            "label": "Bootstrap",
            "diagnostics": {"alpha": 0.05},
            "param_stats": [stats_entry],
            "band": band,
        }

    monkeypatch.setattr(runner, "bootstrap_ci", fake_bootstrap_ci)

    with caplog.at_level(logging.WARNING):
        ok, total = runner.run_batch(
            [str(fail_path), str(ok_path)],
            cfg,
            compute_uncertainty=True,
            unc_method="Bootstrap",
        )

    assert ok == 2 and total == 2
    assert not (tmp_path / "fail_uncertainty.csv").exists()
    assert (tmp_path / "ok_uncertainty.csv").exists()
    assert any("uncertainty failed" in rec.message for rec in caplog.records)
    assert any("Uncertainty failed for" in rec.message for rec in caplog.records)
