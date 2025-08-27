import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from batch import runner
from core.peaks import Peak
from ui.app import pseudo_voigt  # noqa: E402


def test_batch_output_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    x = np.linspace(-5, 5, 101)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    y = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    for i in range(2):
        arr = np.column_stack([x, y])
        np.savetxt(data_dir / f"s{i}.csv", arr, delimiter=",")

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
        "perf_numba": False,
        "perf_gpu": False,
        "perf_cache_baseline": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
        "output_dir": str(out_dir),
        "output_base": "batch",
    }

    runner.run([str(data_dir / "*.csv")], cfg)

    assert (out_dir / "batch_fit.csv").exists()
    assert (out_dir / "batch_uncertainty.csv").exists()
    # traces and bands should also reside in out_dir
    for stem in ["s0", "s1"]:
        assert (out_dir / f"{stem}_trace.csv").exists()
        assert (out_dir / f"{stem}_uncertainty_band.csv").exists()
    # input directory should remain clean
    for f in data_dir.iterdir():
        assert f.suffix == ".csv" and "trace" not in f.name
