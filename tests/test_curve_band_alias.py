import numpy as np
from core import data_io

def test_curve_band_alias_is_normalized():
    res = {"curve_band": (np.arange(5), np.zeros(5), np.ones(5))}
    norm = data_io.normalize_unc_result(res)
    xb, lo, hi = norm["band"]
    assert xb.size == 5 and lo.size == 5 and hi.size == 5
