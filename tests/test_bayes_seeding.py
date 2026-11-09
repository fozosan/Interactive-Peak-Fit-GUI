"""The Bayesian chain is a function of its seed alone (game plan 1.7, review finding 6).

emcee 3 copies numpy's global RandomState when its sampler is built, and its
``random_state`` setter swallows a bad value. ``bayesian_ci`` used to hand it a
RandomState object there, which emcee ignored, and reseeded numpy's global
state before every chunk when Seed-all was on. So ``seed=`` did not control the
chain, and a run moved the global state under everyone else's feet.

Now one seeded ``np.random.default_rng`` stream starts the walkers and another
sets emcee's own generator; numpy's global state is neither read nor written.
"""

import numpy as np
import pytest

from core.uncertainty import bayesian_ci
from infra import performance
from tests.bayes_problem import mos2_problem, recording_sampler

emcee = pytest.importorskip("emcee")


@pytest.fixture(autouse=True)
def _keep_perf_seed(monkeypatch):
    for name in ("_SEED", "_SEED_ALL"):
        monkeypatch.setattr(performance, name, getattr(performance, name))
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _chain(made, kwargs, seed):
    res = bayesian_ci(**kwargs, n_burn=20, n_steps=60, seed=seed)
    return made[-1].get_chain(), res


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


def test_same_seed_same_chain_whatever_the_global_state(monkeypatch):
    made = recording_sampler(monkeypatch)
    kwargs, _ = mos2_problem(1)

    np.random.seed(1)
    a, _ = _chain(made, kwargs, 11)
    np.random.seed(2)
    b, _ = _chain(made, kwargs, 11)
    c, _ = _chain(made, kwargs, 12)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_all_seeds_the_chain_and_leaves_numpy_alone(monkeypatch):
    # With no seed argument, Seed-all's seed is the chain's seed. The run neither
    # reads nor reseeds numpy's global generator.
    made = recording_sampler(monkeypatch)
    kwargs, _ = mos2_problem(1)
    performance.apply_global_seed(5, True)

    np.random.seed(3)
    np.random.random(3)
    before = np.random.get_state()
    a, res = _chain(made, kwargs, None)
    assert _same_state(before, np.random.get_state())
    assert res.diagnostics["seed"] == 5

    np.random.seed(4)
    b, _ = _chain(made, kwargs, None)
    c, _ = _chain(made, kwargs, 6)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
