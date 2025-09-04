import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io  # noqa: E402


def test_bridge_bootstrap_and_bayesian():
    boot = {"type": "bootstrap", "params": {"h1": {"est": 1.0, "sd": 0.2}}}
    res = data_io._ensure_result(boot)
    assert res.method == "bootstrap"
    assert res.method_label == "Bootstrap"

    bayes = {"type": "bayesian", "params": {"h1": {"est": 1.0, "sd": 0.2}}}
    res2 = data_io._ensure_result(bayes)
    assert res2.method == "bayesian"
    assert res2.method_label == "Bayesian (MCMC)"

