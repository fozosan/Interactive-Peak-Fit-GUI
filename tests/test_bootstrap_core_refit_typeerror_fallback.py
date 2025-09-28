import numpy as np

from core import uncertainty as U


def test_core_refit_signature_fallback_and_peaks_injection(monkeypatch):
    n = 10
    theta0 = np.array([10.0, 1.0, 2.0, 0.1])
    residual = np.zeros(n)
    J = np.ones((n, theta0.size))
    x = np.arange(n, dtype=float)
    y = np.zeros(n)

    seen = {"called_without_theta": 0, "called_with_theta": 0, "first_center": None}

    # Simulate a fitter that FAILS if theta_init not provided,
    # and SUCCEEDS (and echoes theta_init) when it is provided.
    def fake_run_fit_consistent(*, x=None, y=None, peaks_in=None, cfg=None,
                                baseline=None, mode=None, fit_mask=None, **kw):
        if "theta_init" not in kw:
            # Force fallback to the "with theta_init" variant
            seen["called_without_theta"] += 1
            raise RuntimeError("need theta_init")
        seen["called_with_theta"] += 1
        th = np.asarray(kw["theta_init"], float).copy()
        # Capture p0 peaks mapping — first peak center must match jittered theta[0]
        if peaks_in and len(peaks_in) >= 1:
            seen["first_center"] = float(peaks_in[0].center)
        return {"theta": th, "fit_ok": True}

    monkeypatch.setattr("core.fit_api.run_fit_consistent", fake_run_fit_consistent)

    # Tiny jitter forces θ→peaks mapping path
    fit_ctx = dict(
        x_all=x, y_all=y,
        residual_fn=lambda th: np.zeros_like(x),
        predict_full=lambda th: np.zeros_like(x),
        strict_refit=True,
        bootstrap_jitter=0.10,   # jitter applied on free params
        lmfit_share_fwhm=False,
        lmfit_share_eta=False,
    )

    res = U.bootstrap_ci(
        theta=theta0,
        residual=residual,
        jacobian=J,
        predict_full=lambda th: np.zeros_like(x),
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=2, seed=42, workers=None, alpha=0.05,
        center_residuals=True, return_band=False,
    )

    d = res.diagnostics or {}
    assert d.get("n_success", 0) >= 1
    # Fallback was exercised at least once
    assert seen["called_without_theta"] >= 1
    assert seen["called_with_theta"] >= 1
    # Peaks injection should mirror jittered theta[0] (not the original theta0[0])
    assert seen["first_center"] is not None
    assert not np.isclose(seen["first_center"], theta0[0])
