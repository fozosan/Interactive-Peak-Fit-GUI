import os
import sys
import pathlib
import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.residuals import build_residual
from infra import performance
from uncertainty import bootstrap as bs

pytestmark = pytest.mark.filterwarnings("ignore:.*")


def test_bootstrap_parallel_deterministic():
    performance.set_seed(123)
    performance.set_max_workers(2)

    x = np.linspace(-1, 1, 20)
    peaks = [Peak(0.0, 1.0, 0.5, 0.2)]
    y_true = performance.eval_total(x, [(1.0, 0.0, 0.5, 0.2)])
    noise = np.linspace(0, 1e-3, x.size)
    y = y_true + noise

    resid_fn = build_residual(x, y, peaks, "add", None, "linear", None)
    theta = np.array([0.0, 1.0, 0.5, 0.2])
    cfg = dict(x=x, y=y, peaks=peaks, mode="add", baseline=None, theta=theta, options={}, n=4, seed=1)

    res1 = bs.bootstrap("classic", cfg, resid_fn)
    res2 = bs.bootstrap("classic", cfg, resid_fn)

    assert np.allclose(res1["params"]["samples"], res2["params"]["samples"])
    performance.set_max_workers(0)

