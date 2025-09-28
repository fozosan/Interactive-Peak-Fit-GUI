import numpy as np

from core import uncertainty as U


def test_fit_mask_kwarg_is_propagated(monkeypatch):
    n = 9
    theta = np.array([1.0, 2.0, 3.0, 0.2])
    residual = np.zeros(n)
    J = np.ones((n, theta.size))
    x = np.arange(n, dtype=float)
    y = np.zeros(n)

    observed = {"fit_mask_seen": False, "mask_seen": False}

    # Signature includes fit_mask; fail if not provided.
    def fake_run_fit_consistent(*, x=None, y=None, peaks_in=None, cfg=None,
                                baseline=None, mode=None, fit_mask=None, **kw):
        observed["fit_mask_seen"] = fit_mask is not None and fit_mask.shape == x.shape
        observed["mask_seen"] = ("mask" in kw)
        return {"theta": np.asarray(theta, float), "fit_ok": True}

    monkeypatch.setattr("core.fit_api.run_fit_consistent", fake_run_fit_consistent)

    res = U.bootstrap_ci(
        theta=theta,
        residual=residual,
        jacobian=J,
        predict_full=lambda th: np.zeros_like(x),
        x_all=x,
        y_all=y,
        fit_ctx={"x": x, "y": y, "strict_refit": True},
        n_boot=2, seed=1, workers=None, alpha=0.05, center_residuals=True, return_band=False,
    )
    assert res is not None
    assert observed["fit_mask_seen"] is True
    assert observed["mask_seen"] is False  # legacy kw must NOT be used
