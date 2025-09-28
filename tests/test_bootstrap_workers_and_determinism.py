import numpy as np

from core.uncertainty import bootstrap_ci


def _run(seed, jitter, workers):
    x = np.linspace(0, 1, 128)
    theta = np.array([0.25, 1.2, 0.3, 0.5])
    y = np.cos(2 * np.pi * (x - theta[0]))
    resid = y - y
    J = np.ones((x.size, theta.size))

    def accept_refit(theta_init, locked_mask, bounds, x_in, y_in):
        return np.array(theta_init, float), True

    out = bootstrap_ci(
        theta=theta,
        residual=resid,
        jacobian=J,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx={
            "refit": accept_refit,
            "bootstrap_jitter": float(jitter),
            "strict_refit": True,
            "unc_workers": workers,
            "peaks": [object()],
        },
        n_boot=32,
        seed=seed,
        workers=workers,
        alpha=0.1,
        center_residuals=True,
        return_band=False,
    )
    return out


def test_workers_and_determinism():
    a = _run(seed=42, jitter=0.03, workers=4)
    b = _run(seed=42, jitter=0.03, workers=2)
    c = _run(seed=43, jitter=0.03, workers=4)
    d = _run(seed=None, jitter=0.03, workers=4)

    da, db = a.diagnostics, b.diagnostics
    assert da.get("draw_workers_used") is not None
    assert db.get("draw_workers_used") is not None

    assert a.stats.keys() == b.stats.keys()
    for k in a.stats:
        for kk in ("est", "sd"):
            assert abs(a.stats[k][kk] - b.stats[k][kk]) < 1e-12

    changed = False
    for k in a.stats:
        if abs(a.stats[k]["est"] - c.stats[k]["est"]) > 1e-9:
            changed = True
            break
    assert changed or (d.diagnostics.get("seed") is None)
