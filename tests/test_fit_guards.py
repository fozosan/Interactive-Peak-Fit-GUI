"""run_fit_consistent refuses fits it cannot do before any solver runs (game plan Phase 0.4).

With no peaks, Modern VP handed scipy.optimize.nnls a design matrix with zero
columns and the interpreter aborted ("double free or corruption"). Auto seeds
on a spectrum with no peaks produce exactly that, so a batch died with it. That
case runs in a subprocess here, so a regression cannot take pytest down too.
"""
import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from core import fit_api, models
from core.peaks import Peak

REPO = Path(__file__).resolve().parents[1]

ZERO_PEAKS = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import numpy as np
    from batch import runner
    from core import fit_api, models
    from core.peaks import Peak

    x = np.linspace(0.0, 100.0, 201)
    no_peaks = np.full_like(x, 10.0)  # flat: auto seeding finds nothing
    window = (x >= 20.0) & (x <= 80.0)
    cfg = {"solver": "modern_vp", "modern_vp": {"centers_in_window": True},
           "baseline": {"method": "als"}}
    res = fit_api.run_fit_consistent(x, no_peaks, [], cfg, np.full_like(x, 10.0), "add", window)
    print("direct", res["fit_ok"], res["skip_reason"], flush=True)
    """
)


def test_zero_peaks_exits_cleanly(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", ZERO_PEAKS, str(in_dir), str(out_dir)],
        cwd=REPO, env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=300,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "direct False no_peaks" in proc.stdout


def _two_peak_case():
    x = np.linspace(0.0, 100.0, 201)
    y = models.pv_sum(x, [Peak(40.0, 100.0, 5.0, 0.5), Peak(60.0, 80.0, 6.0, 0.5)]) + 10.0
    cfg = {"solver": "modern_vp", "modern_vp": {"centers_in_window": True}}
    return x, y, cfg, np.full_like(x, 10.0)


def test_a_window_with_no_more_points_than_free_parameters_is_not_fitted():
    x, y, cfg, baseline = _two_peak_case()
    window = np.zeros_like(x, bool)
    window[78:84] = True  # 6 points, 39.0 to 41.5
    seeds = [Peak(40.0, 90.0, 4.5, 0.5), Peak(60.0, 70.0, 5.5, 0.5)]  # 6 free parameters

    res = fit_api.run_fit_consistent(x, y, seeds, cfg, baseline, "add", window)

    assert res["fit_ok"] is False
    assert res["skip_reason"] == "too_few_points"
    assert "6 points" in res["message"] and "6 free parameters" in res["message"]
    assert [p.center for p in res["peaks_out"]] == [40.0, 60.0]

    # Locked centres leave 4 free parameters, so the same 6 points are enough.
    locked = [Peak(40.0, 90.0, 4.5, 0.5, lock_center=True),
              Peak(60.0, 70.0, 5.5, 0.5, lock_center=True)]
    res = fit_api.run_fit_consistent(x, y, locked, cfg, baseline, "add", window)
    assert res["skip_reason"] is None


def test_an_empty_window_is_not_fitted():
    x, y, cfg, baseline = _two_peak_case()
    seeds = [Peak(40.0, 90.0, 4.5, 0.5)]

    res = fit_api.run_fit_consistent(x, y, seeds, cfg, baseline, "add", np.zeros_like(x, bool))

    assert res["fit_ok"] is False
    assert res["skip_reason"] == "too_few_points"
    assert res["message"] == "no points in the fit window"


def test_a_normal_fit_reports_no_skip_reason():
    x, y, cfg, baseline = _two_peak_case()
    seeds = [Peak(41.0, 90.0, 4.5, 0.5), Peak(59.0, 70.0, 5.5, 0.5)]

    res = fit_api.run_fit_consistent(x, y, seeds, cfg, baseline, "add", np.ones_like(x, bool))

    assert res["fit_ok"] is True
    assert res["skip_reason"] is None
    assert [round(p.center, 3) for p in res["peaks_out"]] == [40.0, 60.0]
