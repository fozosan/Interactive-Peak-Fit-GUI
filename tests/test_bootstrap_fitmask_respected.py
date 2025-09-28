import numpy as np

from core import peaks
from core import uncertainty as U
from core import fit_api


def test_bootstrap_respects_fit_mask(monkeypatch):
    x = np.linspace(0, 10, 101)
    baseline = np.zeros_like(x)
    y = np.sin(x) + baseline
    pk = peaks.Peak(5.0, 1.0, 1.0, 0.5)
    theta0 = np.array([pk.center, pk.height, pk.fwhm, pk.eta], float)
    residual = np.zeros_like(x)
    jac = np.zeros((x.size, theta0.size), float)

    fit_mask = (x >= 3) & (x <= 7)
    seen_masks = []

    def fake_run_fit_consistent(*args, **kwargs):
        fit_mask_val = kwargs.get("fit_mask")
        if fit_mask_val is None and len(args) >= 7:
            fit_mask_val = args[6]
        if fit_mask_val is not None:
            seen_masks.append(np.asarray(fit_mask_val, bool))
        theta_init = kwargs.get("theta_init", theta0)
        return {
            "theta": np.asarray(theta_init, float),
            "fit_ok": True,
        }

    from inspect import Signature, Parameter

    fake_run_fit_consistent.__signature__ = Signature(
        [
            Parameter("x_in", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("y_in", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("peaks_in", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("cfg", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("baseline", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("mode", Parameter.POSITIONAL_OR_KEYWORD, default="add"),
            Parameter("fit_mask", Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
    )

    monkeypatch.setattr(fit_api, "run_fit_consistent", fake_run_fit_consistent)

    fit_ctx = {
        "peaks": [pk],
        "baseline": baseline,
        "mode": "add",
        "theta0": theta0,
        "x_all": x,
        "y_all": y,
        "fit_mask": fit_mask,
        "strict_refit": True,
    }

    U.bootstrap_ci(
        theta=theta0,
        residual=residual,
        jacobian=jac,
        predict_full=lambda th: y,
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        n_boot=3,
        seed=123,
        workers=None,
        jitter=0.0,
    )

    assert seen_masks, "refit should have been invoked"
    for mask in seen_masks:
        assert mask.shape == x.shape
        assert np.array_equal(mask, fit_mask)
