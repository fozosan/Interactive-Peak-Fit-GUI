import numpy as np
from core.peaks import Peak
from fit.bounds import pack_theta_bounds


def test_pack_theta_bounds_respects_per_peak_caps():
    x = np.linspace(0, 100, 501)
    peaks = [Peak(40.0, 5.0, 12.0, 0.5), Peak(60.0, 3.0, 8.0, 0.4)]
    opts = {"centers_in_window": True, "width_caps": [5.0, None]}
    theta, (lo, hi) = pack_theta_bounds(peaks, x, opts)
    assert hi[2] <= 5.0 + 1e-12        # first peak width upper bound
    assert np.isfinite(hi[6]) and hi[6] > 5.0
