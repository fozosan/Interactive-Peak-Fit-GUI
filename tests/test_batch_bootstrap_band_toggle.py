import numpy as np
import pytest
from pathlib import Path

from batch import runner
from core.peaks import Peak
from ui.app import pseudo_voigt


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    x = np.linspace(-5.0, 5.0, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    path = tmp_path / "a.csv"
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")
    return path


def _install_bootstrap_stub(monkeypatch):
    seen: list[tuple[bool, object]] = []

    def fake_bootstrap_ci(**kwargs):
        rb = bool(kwargs.get("return_band"))
        workers_val = kwargs.get("workers")
        assert workers_val != 0
        seen.append((rb, workers_val))
        xb = np.linspace(0.0, 1.0, 5)
        lo = np.full_like(xb, -1.0)
        hi = np.full_like(xb, 1.0)
        stats = {
            "center_est": 0.0,
            "center_sd": 0.1,
            "center_p2_5": -0.2,
            "center_p97_5": 0.2,
            "height_est": 1.0,
            "height_sd": 0.1,
            "height_p2_5": 0.8,
            "height_p97_5": 1.2,
            "fwhm_est": 1.0,
            "fwhm_sd": 0.1,
            "fwhm_p2_5": 0.8,
            "fwhm_p97_5": 1.2,
            "eta_est": 0.5,
            "eta_sd": 0.05,
            "eta_p2_5": 0.4,
            "eta_p97_5": 0.6,
        }
        return {
            "label": "Bootstrap",
            "method": "bootstrap",
            "band": (xb, lo, hi),
            "param_stats": [stats],
            "diagnostics": {"alpha": kwargs.get("alpha", 0.05)},
            "rmse": 0.0,
            "dof": 1,
        }

    monkeypatch.setattr(runner, "bootstrap_ci", fake_bootstrap_ci)
    monkeypatch.setattr(runner.performance, "apply_global_seed", lambda *a, **k: None)
    return seen


def _base_config(tmp_dir: Path) -> dict:
    return {
        "peaks": [
            {
                "center": 0.0,
                "height": 1.0,
                "fwhm": 1.0,
                "eta": 0.5,
            }
        ],
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
        "output_dir": str(tmp_dir),
        "output_base": "batch",
        "bootstrap_n": 5,
        "bootstrap_jitter": 0.0,
        "unc_workers": 0,
        "unc_band_workers": 0,
        "perf_parallel_strategy": "outer",
        "perf_blas_threads": 0,
        "export_unc_wide": False,
    }


def test_bootstrap_band_file_respects_toggle_on(monkeypatch, tmp_path: Path, sample_csv: Path):
    seen = _install_bootstrap_stub(monkeypatch)
    out_dir = tmp_path / "out_on"
    out_dir.mkdir()
    cfg = _base_config(out_dir)
    cfg["ui_band_pref_bootstrap"] = True
    ok, total = runner.run_batch([str(sample_csv)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    assert (ok, total) == (1, 1)
    assert len(seen) == 1 and seen[0][0] is True
    band_path = out_dir / "a_band.csv"
    assert band_path.exists()
    assert (out_dir / "a_uncertainty.csv").exists()


def test_bootstrap_band_file_respects_toggle_off(monkeypatch, tmp_path: Path, sample_csv: Path):
    seen = _install_bootstrap_stub(monkeypatch)
    out_dir = tmp_path / "out_off"
    out_dir.mkdir()
    cfg = _base_config(out_dir)
    cfg["ui_band_pref_bootstrap"] = False
    ok, total = runner.run_batch([str(sample_csv)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    assert (ok, total) == (1, 1)
    assert len(seen) == 1 and seen[0][0] is False
    band_path = out_dir / "a_band.csv"
    assert not band_path.exists()
    assert (out_dir / "a_uncertainty.csv").exists()
