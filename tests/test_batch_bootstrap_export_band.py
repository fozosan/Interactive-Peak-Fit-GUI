import numpy as np

from core.uncertainty import bootstrap_ci


def test_batch_bootstrap_produces_band_when_requested():
    x = np.linspace(0, 1, 64)
    y = np.sin(6 * x)
    theta = np.array([1.0, 2.0, 3.0, 4.0])

    def ymodel_fn(t):
        assert t is not None
        return y

    res = bootstrap_ci(
        theta=theta,
        residual=np.zeros_like(y),
        jacobian=np.zeros((x.size, theta.size)),
        predict_full=ymodel_fn,
        x_all=x,
        y_all=y,
        fit_ctx={"unc_workers": 0, "unc_band_workers": 0, "bootstrap_jitter": 0.0},
        n_boot=32,
        workers=0,
        return_band=True,
    )
    assert res.band is not None
    bx, blo, bhi = res.band
    assert bx.shape == blo.shape == bhi.shape == x.shape
