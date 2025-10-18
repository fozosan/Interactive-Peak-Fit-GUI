import numpy as np

from core.uncertainty import bootstrap_ci


def test_bootstrap_band_no_unbound():
    """Bootstrap CI band should not raise UnboundLocalError when gated."""
    x = np.linspace(0, 1, 128)
    theta = np.array([10.0, 5.0, 3.0, 0.5])
    y0 = np.sin(6 * x) + 0.05 * np.random.RandomState(0).randn(x.size)
    residual = np.zeros_like(y0)

    def predict_full(_theta):
        return y0

    result = bootstrap_ci(
        theta=theta,
        residual=residual,
        jacobian=np.zeros((x.size, theta.size)),
        predict_full=predict_full,
        x_all=x,
        y_all=y0,
        fit_ctx={"unc_workers": 0, "unc_band_workers": 0, "bootstrap_jitter": 0.0},
        n_boot=16,
        workers=0,
        return_band=True,
    )

    if result.band is not None:
        band_x, band_lo, band_hi = result.band
        assert band_x.shape == band_lo.shape == band_hi.shape
