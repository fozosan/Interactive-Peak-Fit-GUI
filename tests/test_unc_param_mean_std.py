import math
import numpy as np
from core import data_io


def test_normalize_param_mean_std():
    mean = np.arange(8, dtype=float)
    std = np.full(8, 0.1)
    unc = {"param_mean": mean, "param_std": std}
    norm = data_io.normalize_unc_result(unc)
    assert len(norm["stats"]) == 2
    for row in norm["stats"]:
        for pname in ("center", "height", "fwhm", "eta"):
            blk = row[pname]
            assert math.isfinite(blk.get("est", float("nan")))
            assert math.isfinite(blk.get("sd", float("nan")))
            assert math.isfinite(blk.get("p2_5", float("nan")))
            assert math.isfinite(blk.get("p97_5", float("nan")))
