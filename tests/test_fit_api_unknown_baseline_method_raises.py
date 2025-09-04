import numpy as np
import pytest
from core import fit_api, peaks


def test_fit_api_unknown_baseline_method_raises():
    x = np.linspace(-1, 1, 11)
    y = np.zeros_like(x)
    seeds = [peaks.Peak(0.0, 1.0, 1.0, 0.5)]
    cfg = {"baseline": {"method": "nope"}}
    mask = np.ones_like(x, bool)
    with pytest.raises(ValueError, match="Unknown baseline method"):
        fit_api.run_fit_consistent(x, y, seeds, cfg, None, "add", mask)
