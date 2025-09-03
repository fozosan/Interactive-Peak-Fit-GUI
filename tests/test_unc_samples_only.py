import math
import numpy as np
from core import data_io


def test_normalize_samples_only():
    unc = {"label": "Bootstrap (residual)", "params": {"samples": np.random.randn(50, 8)}}
    norm = data_io.normalize_unc_result(unc)
    assert len(norm["stats"]) == 2
    for row in norm["stats"]:
        for pname in ("center", "height", "fwhm", "eta"):
            blk = row[pname]
            assert math.isfinite(blk.get("est", float("nan")))
            assert math.isfinite(blk.get("sd", float("nan")))
