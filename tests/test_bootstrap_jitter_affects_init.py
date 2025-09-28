import numpy as np

from core.uncertainty import bootstrap_ci


def _run_with_jitter(jitter):
    x = np.linspace(0, 1, 32)
    theta = np.array([0.2, 1.1, 0.3, 0.4])
    y = np.cos(2 * np.pi * (x - theta[0]))
    resid = y - y
    J = np.ones((x.size, theta.size))

    def accept(theta_init, locked_mask, bounds, x_in, y_in):
        return np.array(theta_init, float), True

    out = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx={
            "refit": accept,
            "bootstrap_jitter": float(jitter),
            "strict_refit": True,
            "peaks": [object()],
        },
        n_boot=16,
        seed=321,
        workers=None,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    return out


def test_jitter_changes_diagnostics_and_draws():
    baseline = _run_with_jitter(0.0)
    jittered = _run_with_jitter(0.1)

    assert abs(baseline.diagnostics.get("jitter_last_rms", 0.0)) < 1e-12
    assert jittered.diagnostics.get("jitter_last_rms", 0.0) > 0.0

    keys = set(baseline.stats.keys()) & set(jittered.stats.keys())
    assert keys
    diffs = [abs(baseline.stats[k]["est"] - jittered.stats[k]["est"]) for k in keys]
    assert any(d > 1e-8 for d in diffs)
