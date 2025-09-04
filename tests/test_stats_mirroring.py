from core.data_io import normalize_unc_result


def test_stats_mirrored_keys():
    payload = {
        "method": "asymptotic",
        "stats": [
            {
                "index": 1,
                "center": {
                    "est": 1.0,
                    "sd": 0.1,
                    "ci_lo": 0.8,
                    "ci_hi": 1.2,
                    "p2_5": 0.8,
                    "p97_5": 1.2,
                },
                "height": {
                    "est": 5.0,
                    "sd": 0.5,
                    "ci_lo": 4.0,
                    "ci_hi": 6.0,
                    "p2_5": 4.0,
                    "p97_5": 6.0,
                },
                "fwhm": {
                    "est": 2.0,
                    "sd": 0.2,
                    "ci_lo": 1.6,
                    "ci_hi": 2.4,
                    "p2_5": 1.6,
                    "p97_5": 2.4,
                },
                "eta": {
                    "est": 0.3,
                    "sd": 0.05,
                    "ci_lo": 0.2,
                    "ci_hi": 0.4,
                    "p2_5": 0.2,
                    "p97_5": 0.4,
                },
            }
        ],
    }
    norm = normalize_unc_result(payload)
    assert "param_stats" in norm
    assert "stats" in norm
    for k in ("center", "height", "fwhm", "eta"):
        assert k in norm["param_stats"]

