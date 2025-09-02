import numpy as np
import pandas as pd
from batch import runner
from core.peaks import Peak


def test_batch_wide_toggle(tmp_path):
    x = np.linspace(-3, 3, 181)
    y = np.exp(-(x ** 2))
    p = tmp_path / "a.csv"
    np.savetxt(p, np.column_stack([x, y]), delimiter=",")

    pk = Peak(0.0, 1.0, 0.8, 0.3)

    cfg = {
        "peaks": [pk.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "output_dir": str(tmp_path),
        "output_base": "batch",
        "baseline_uses_fit_range": True,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
    }

    # off
    ok, _ = runner.run_batch(
        [str(p)], cfg | {"export_unc_wide": False}, compute_uncertainty=True, unc_method="Asymptotic"
    )
    assert ok == 1
    assert not (tmp_path / "a_uncertainty_wide.csv").exists()

    # on
    ok, _ = runner.run_batch(
        [str(p)], cfg | {"export_unc_wide": True}, compute_uncertainty=True, unc_method="Asymptotic"
    )
    assert ok == 1
    assert (tmp_path / "a_uncertainty_wide.csv").exists()

