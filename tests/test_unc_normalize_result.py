import pathlib
import sys

import numpy as np
import pytest

# allow "core" imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.data_io import normalize_unc_result


def test_normalize_result_arrays_and_band():
    """normalize_unc_result should coerce arrays and map bands."""
    raw = {
        "method": "bootstrap",
        "param_stats": {
            "center": {"est": np.array([1.0]), "sd": np.array([0.1]), "p97_5": np.array([1.1])},
            "height": {"est": np.array([2.0]), "sd": np.array([0.2]), "p2_5": np.array([1.8])},
            "fwhm": {"est": np.array([0.5]), "sd": np.array([0.05])},
            "eta": {"est": np.array([0.1]), "sd": np.array([0.01])},
        },
        "curve_band": [np.array([0.0, 1.0]), np.array([-1.0, -0.5]), np.array([1.0, 1.5])],
        "n_boot": 10,
    }

    out = normalize_unc_result(raw)
    # method resolved to canonical label
    assert out["label"] == "Bootstrap (residual)"
    # band becomes lists for json friendliness
    assert isinstance(out["band"], (list, tuple))
    xb, lob, hib = out["band"]
    assert isinstance(xb, list) and xb == [0.0, 1.0]
    # stats present with floats and percentiles
    stats0 = out["stats"][0]
    assert stats0["center"]["sd"] == pytest.approx(0.1)
    assert stats0["height"]["p2_5"] == pytest.approx(1.8)
    assert stats0["center"]["p97_5"] == pytest.approx(1.1)


def test_normalize_infers_bayesian_from_backend():
    """Even without an explicit label, backend hints infer the method."""
    raw = {
        "backend": "emcee",
        "samples": np.zeros((5, 2)),
        "param_stats": {
            "center": {"est": [1.0], "sd": [0.1], "p2_5": [0.9]},
            "height": {"est": [2.0], "sd": [0.2], "p97_5": [2.4]},
            "fwhm": {"est": [0.5], "sd": [0.05]},
            "eta": {"est": [0.1], "sd": [0.01]},
        },
    }

    out = normalize_unc_result(raw)
    assert out["label"] == "Bayesian (MCMC)"
    assert out["n_draws"] == 5
    stats0 = out["stats"][0]
    # carries percentiles when provided
    assert stats0["center"]["p2_5"] == pytest.approx(0.9)
    assert stats0["height"]["p97_5"] == pytest.approx(2.4)

