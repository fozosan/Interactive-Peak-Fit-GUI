import numpy as np
import pytest
from core.data_io import normalize_unc_result


def _mk_param_stats_two_peaks():
    return {
        "center": {"est": [1.0, 2.0], "sd": [0.01, 0.02]},
        "height": {"est": [10.0, 20.0], "sd": [0.5, 1.0]},
        "fwhm": {"est": [0.8, 1.2], "sd": [0.03, 0.04]},
        "eta": {"est": [0.5, 0.2], "sd": [0.02, 0.01]},
    }


def test_normalize_unc_result_builds_per_peak_rows():
    inp = {"method": "asymptotic", "param_stats": _mk_param_stats_two_peaks()}
    out = normalize_unc_result(inp)
    assert isinstance(out, dict)
    assert isinstance(out.get("stats"), list)
    assert len(out["stats"]) == 2
    p1 = out["stats"][0]
    assert set(p1.keys()) >= {"center", "height", "fwhm", "eta"}
    for k in ("center", "height", "fwhm", "eta"):
        assert "est" in p1[k]


def test_normalize_unc_result_handles_numpy_and_missing_label():
    ps = {
        "center": {
            "est": None,
            "value": None,
            "mean": None,
            "median": np.array([1.0, 2.0]),
            "sd": None,
            "stderr": None,
            "sigma": np.array([0.1, 0.2]),
        },
        "height": {"est": [5.0, 6.0], "sd": [0.3, 0.4]},
        "fwhm": {
            "est": None,
            "value": None,
            "mean": None,
            "median": np.array([0.7, 0.9]),
            "sd": None,
            "stderr": None,
            "sigma": np.array([0.05, 0.06]),
        },
        "eta": {"est": [0.2, 0.3], "sd": None, "stderr": None, "sigma": np.array([0.01, 0.02])},
    }
    inp = {"method": "bootstrap", "param_stats": ps}
    out = normalize_unc_result(inp)
    assert out.get("method") == "bootstrap"
    assert out.get("label") == "Bootstrap (residual)"
    assert isinstance(out.get("stats"), list)
    assert len(out["stats"]) == 2
    for row in out["stats"]:
        for k in ("center", "height", "fwhm", "eta"):
            assert isinstance(row[k]["est"], (int, float))
            if "sd" in row[k] and row[k]["sd"] is not None:
                assert isinstance(row[k]["sd"], (int, float))


def test_normalize_unc_result_safe_defaults_on_empty():
    out = normalize_unc_result({})
    assert out.get("label") == "unknown" or out.get("label") in {None, ""}
    assert isinstance(out.get("stats"), list)
