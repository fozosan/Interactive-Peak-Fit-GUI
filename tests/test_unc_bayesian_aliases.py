import math
import numpy as np

from core.data_io import normalize_unc_result


def test_normalize_bayesian_quantile_aliases():
    stats = {
        "center": {"est": [1.0], "sd": [0.1], "p2.5": [0.9], "p97.5": [1.1]},
        "height": {"est": [2.0], "sd": [0.2], "p2.5": [1.8], "p97.5": [2.2]},
        "fwhm": {"est": [0.5], "sd": [0.05], "p2.5": [0.4], "p97.5": [0.6]},
        "eta": {"est": [0.3], "sd": [0.03], "p2.5": [0.24], "p97.5": [0.36]},
    }
    out = normalize_unc_result({"param_stats": stats})
    assert len(out["stats"]) == 1
    row = out["stats"][0]
    for pname in ("center", "height", "fwhm", "eta"):
        blk = row[pname]
        assert math.isfinite(blk.get("p2_5", float("nan")))
        assert math.isfinite(blk.get("p97_5", float("nan")))


def test_normalize_bayesian_posterior_mean_std_only():
    mean = np.arange(4, dtype=float)
    std = np.full(4, 0.1)
    out = normalize_unc_result({"posterior": {"mean": mean, "std": std}})
    assert len(out["stats"]) == 1
    row = out["stats"][0]
    for pname in ("center", "height", "fwhm", "eta"):
        blk = row[pname]
        assert math.isfinite(blk.get("est", float("nan")))
        assert math.isfinite(blk.get("sd", float("nan")))
        assert math.isfinite(blk.get("p2_5", float("nan")))
        assert math.isfinite(blk.get("p97_5", float("nan")))

