"""Uncertainty workers set to 0 ("auto") run sequentially, without a thread pool.

Auto used to mean one thread per CPU. The draws hold the GIL, so the pools never
sped anything up, and 24 threads was the slowest setting measured: bootstrap
n=200 took 4.6 s against 3.4 s sequential, Bayesian 14.8 s against 6.6 s.
A pool of one thread is sequential too, without the per-task overhead.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from batch.runner import run_batch
from core import uncertainty as unc
from infra import performance
from tests.conftest import bayes_knobs, ensure_unc_common


@pytest.fixture(autouse=True)
def _restore_seed_state(monkeypatch):
    # run_batch seeds the global RNGs and stores the seed in infra.performance.
    for name in ("_SEED", "_SEED_ALL", "_UNC_WORKERS"):
        monkeypatch.setattr(performance, name, getattr(performance, name))
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def pools(monkeypatch):
    """Record the max_workers of every thread pool core.uncertainty starts."""
    created = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, *args, **kwargs):
            created.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    monkeypatch.setattr(unc, "ThreadPoolExecutor", RecordingPool)
    return created


@pytest.fixture
def spectrum(tmp_path):
    x = np.linspace(0.0, 100.0, 201)
    rng = np.random.default_rng(7)
    y = 50.0 + 0.02 * x + 4.0 / (1.0 + 4.0 * ((x - 40.0) / 6.0) ** 2) + rng.normal(0.0, 0.15, x.size)
    path = tmp_path / "one.csv"
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")
    return path


def _run_batch(path, out_dir, method, workers):
    cfg = {
        "peaks": [{"center": 40.0, "height": 4.0, "fwhm": 6.0, "eta": 0.5}],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(out_dir),
        "perf_seed_all": True,
        "perf_seed": 1234,
        "bootstrap_n": 8,
        "unc_boot_solver": "modern_vp",
    }
    cfg.update(ensure_unc_common({}))
    cfg.update(bayes_knobs(walkers=0, burn=20, steps=40, thin=1))
    if workers is not None:
        cfg["perf_unc_workers"] = workers
    ok, total = run_batch([str(path)], cfg, compute_uncertainty=True, unc_method=method)
    assert (ok, total) == (1, 1)
    assert (out_dir / f"{path.stem}_uncertainty.csv").exists()


def test_auto_resolves_to_one_worker():
    performance.set_unc_workers(0)
    assert performance.get_parallel_config().unc_workers == 1

    performance.set_unc_workers(3)
    assert performance.get_parallel_config().unc_workers == 3


@pytest.mark.parametrize("method", ["bootstrap", "bayesian"])
def test_batch_auto_workers_start_no_pool(method, pools, spectrum, tmp_path):
    if method == "bayesian":
        pytest.importorskip("emcee")
    performance.set_unc_workers(0)  # what the GUI applies for "0 = auto"

    _run_batch(spectrum, tmp_path / "out", method, workers=None)

    assert pools == []


@pytest.mark.parametrize("method", ["bootstrap", "bayesian"])
def test_batch_explicit_workers_still_get_a_pool(method, pools, spectrum, tmp_path):
    # Shows the spy sees pools: an explicit count above one still runs in parallel.
    if method == "bayesian":
        pytest.importorskip("emcee")
    performance.set_unc_workers(0)

    _run_batch(spectrum, tmp_path / "out", method, workers=2)

    assert 2 in pools
