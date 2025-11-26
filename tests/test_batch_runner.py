import numpy as np
import pytest
from pathlib import Path

from batch.runner import run_batch
from core.peaks import Peak
from tests.conftest import bootstrap_cfg, ensure_unc_common
from ui.app import pseudo_voigt


def _write_sample(tmp_path: Path) -> tuple[Path, Peak]:
    x = np.linspace(-5.0, 5.0, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    file_path = tmp_path / "sample.csv"
    np.savetxt(file_path, np.column_stack([x, y]), delimiter=",")
    return file_path, peak


def _base_batch_cfg(tmp_path: Path, peak: Peak) -> dict:
    cfg = {
        "peaks": [
            {
                "center": peak.center,
                "height": peak.height,
                "fwhm": peak.fwhm,
                "eta": peak.eta,
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
        "output_dir": str(tmp_path),
        "output_base": "batch",
    }
    return ensure_unc_common(cfg)


def test_run_batch_bootstrap_requires_explicit_samples(tmp_path: Path):
    datafile, peak = _write_sample(tmp_path)
    cfg = _base_batch_cfg(tmp_path, peak)
    cfg.update(bootstrap_cfg(n=200, jitter=0.0))

    ok, total = run_batch([str(datafile)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    assert (ok, total) == (1, 1)


def test_run_batch_bootstrap_missing_samples_errors(tmp_path: Path):
    datafile, peak = _write_sample(tmp_path)
    cfg = _base_batch_cfg(tmp_path, peak)

    with pytest.raises(KeyError, match="bootstrap_n"):
        run_batch([str(datafile)], cfg, compute_uncertainty=True, unc_method="bootstrap")
