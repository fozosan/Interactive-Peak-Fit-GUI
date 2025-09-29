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

