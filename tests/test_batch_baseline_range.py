import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from batch import runner
from core.peaks import Peak
from core import signals
from ui.app import pseudo_voigt  # noqa: E402


def test_batch_baseline_range(tmp_path):
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    baseline_true = 0.1 * x
    y = baseline_true + pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    data_file = tmp_path / "s.csv"
    np.savetxt(data_file, np.column_stack([x, y]), delimiter=",")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = {
        "peaks": [peak.__dict__],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": True,
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        "classic": {},
        "baseline_uses_fit_range": True,
        "fit_xmin": -1.0,
        "fit_xmax": 1.0,
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
        "output_dir": str(out_dir),
        "output_base": "batch",
    }
    runner.run([str(data_file)], cfg)
    trace_df = pd.read_csv(out_dir / "s_trace.csv")
    baseline_col = trace_df["baseline"].to_numpy()
    mask = (x >= -1.0) & (x <= 1.0)
    z_sub = signals.als_baseline(y[mask], lam=1e5, p=0.001, niter=10, tol=0.0)
    expected = np.interp(x, x[mask], z_sub, left=z_sub[0], right=z_sub[-1])
    assert np.allclose(baseline_col, expected)
