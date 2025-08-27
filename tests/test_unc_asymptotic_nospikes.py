import numpy as np
from core import fit_api, uncertainty


def test_unc_asymptotic_nospikes(two_peak_data):
    res = fit_api.run_fit_consistent(**two_peak_data, return_jacobian=True)
    unc = uncertainty.asymptotic_ci(
        res["theta"], res["residual_fn"], res["jacobian"], res["ymodel_fn"],
        alpha=0.05, svd_rcond=1e-10, grad_mode="complex"
    )
    x, lo, hi = unc["band"]
    width = hi - lo
    assert np.all(np.isfinite(width))
    assert np.all(width >= 0)
    if width.size > 4:
        core = width[1:-1]
        jump = np.abs(np.diff(core))
        med = np.median(jump)
        if med == 0:
            assert np.max(jump) <= 1e-12
        else:
            assert np.max(jump) <= 5 * med
