import numpy as np

from core.models import pv_sum
from core.peaks import Peak
from fit import modern_vp


def test_modern_vp_iterate_respects_bounds_and_reference():
    x = np.linspace(0.0, 60.0, 400)
    expected_theta = np.array([20.0, 5.0, 5.0, 0.5, 40.0, 2.0, 6.0, 0.3], dtype=float)
    expected_peaks = [
        Peak(expected_theta[0], expected_theta[1], expected_theta[2], expected_theta[3]),
        Peak(expected_theta[4], expected_theta[5], expected_theta[6], expected_theta[7]),
    ]
    y = pv_sum(x, expected_peaks)
    start = [
        Peak(19.0, 4.0, 6.0, 0.5),
        Peak(41.0, 1.5, 7.0, 0.3),
    ]

    state = modern_vp.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(10):
        state, _, _, _, _ = modern_vp.iterate(state)

    theta_out = np.asarray(state["theta"], dtype=float)

    # New VP enforces bounds and numeric hygiene; compare within tolerance:
    assert np.all(np.isfinite(theta_out))
    assert np.all(theta_out[2::4] > 0.0)  # FWHM > 0
    assert np.all((theta_out[3::4] >= 0.0) & (theta_out[3::4] <= 1.0))  # 0 ≤ η ≤ 1
    assert np.allclose(theta_out, expected_theta, rtol=1e-3, atol=1e-3)
