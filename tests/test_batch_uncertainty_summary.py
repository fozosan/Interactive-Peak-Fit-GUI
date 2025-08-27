import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from batch import runner
from core.peaks import Peak
from ui.app import pseudo_voigt  # noqa: E402


def test_batch_uncertainty_summary(tmp_path):
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    for i in range(2):
        arr = np.column_stack([x, y])
        np.savetxt(tmp_path / f"s{i}.csv", arr, delimiter=",")

    cfg = {
        "peaks": [peak.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "peak_output": str(tmp_path / "batch_fit.csv"),
        "unc_output": str(tmp_path / "batch_uncertainty.csv"),
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        "classic": {},
        "baseline_uses_fit_range": True,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
    }

    runner.run([str(tmp_path / "*.csv")], cfg)

    fit_df = pd.read_csv(tmp_path / "batch_fit.csv")
    for col in [
        "solver_choice",
        "solver_loss",
        "solver_weight",
        "solver_fscale",
        "solver_maxfev",
        "solver_restarts",
        "solver_jitter_pct",
        "baseline_uses_fit_range",
        "perf_numba",
    ]:
        assert col in fit_df.columns

    unc_df = pd.read_csv(tmp_path / "batch_uncertainty.csv")
    assert set(unc_df.columns) == {"file", "peak", "param", "value", "ci_lo", "ci_hi", "method", "rmse", "dof"}
    assert len(unc_df) == 2 * 1 * 4
