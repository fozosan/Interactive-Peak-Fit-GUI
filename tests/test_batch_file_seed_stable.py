"""Per-file batch seeds are the same on every run.

With Seed all on, each file's fit restarts are seeded per file. That seed came
from Python's ``hash()``, which is salted per process, so the same batch with
the same seed fitted differently on every run: 4 restarts with 10 % jitter moved
theta by 0.2-1.0 between processes on a real MoS2 spectrum. On the spectra
below the two peaks even come out with their labels swapped. The bootstrap
refits restarted from unseeded jitter as well, so the uncertainty differed
between runs even once the fit agreed.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from batch import runner
from core import fit_api
from infra import performance

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_seed_state(monkeypatch):
    # run_batch seeds the global RNGs and stores the seed in infra.performance.
    for name in ("_SEED", "_SEED_ALL"):
        monkeypatch.setattr(performance, name, getattr(performance, name))
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# Runs one batch in a fresh interpreter and prints the rng_seed of every fit.
_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from batch import runner
from core import fit_api
seeds = []
_fit = fit_api.run_fit_consistent
def _spy(*args, **kwargs):
    seeds.append(kwargs.get("rng_seed"))
    return _fit(*args, **kwargs)
fit_api.run_fit_consistent = _spy
runner.run_batch([sys.argv[3]], json.loads(sys.argv[2]), compute_uncertainty=True, unc_method="bootstrap")
print(json.dumps(seeds))
"""


def _write_spectra(folder):
    folder.mkdir()
    x = np.linspace(0.0, 100.0, 201)
    for k, name in enumerate(["a.csv", "b.csv"]):
        rng = np.random.default_rng(10 + k)
        y = (
            50.0
            + 0.02 * x
            + 4.0 / (1.0 + 4.0 * ((x - 40.0) / 6.0) ** 2)
            + 2.5 * np.exp(-4.0 * np.log(2.0) * ((x - 50.0) / 7.0) ** 2)
            + rng.normal(0.0, 0.15, x.size)
        )
        np.savetxt(folder / name, np.column_stack([x, y]), delimiter=",")


def _config(out_dir, base_seed):
    return {
        # Two overlapping peaks, so restarts from jittered starts matter.
        "peaks": [
            {"center": 38.0, "height": 4.0, "fwhm": 6.0, "eta": 0.5},
            {"center": 52.0, "height": 2.0, "fwhm": 6.0, "eta": 0.5},
        ],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(out_dir),
        "perf_seed_all": True,
        "perf_seed": base_seed,
        "solver_restarts": 4,
        "solver_jitter_pct": 10.0,
        "bootstrap_n": 16,
        "unc_boot_solver": "modern_vp",
        "perf_unc_workers": 1,
    }


def test_same_seed_gives_identical_bootstrap_output_on_every_run(tmp_path):
    data = tmp_path / "data"
    _write_spectra(data)
    seeds, outputs = [], []
    for salt in ("1", "2"):  # two hash salts stand in for two separate runs
        out_dir = tmp_path / f"run{salt}"
        env = dict(os.environ, PYTHONHASHSEED=salt, MPLBACKEND="Agg")
        env["HOME"] = env["USERPROFILE"] = str(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, str(REPO), json.dumps(_config(out_dir, 1234)), str(data / "*.csv")],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        seeds.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        outputs.append({p.name: p.read_bytes() for p in sorted(out_dir.iterdir())})

    assert seeds[0] == seeds[1]
    assert {"a_uncertainty.csv", "b_uncertainty.csv"} <= outputs[0].keys()
    assert outputs[0].keys() == outputs[1].keys()
    for name, content in outputs[0].items():
        assert content == outputs[1][name], f"{name} differs between runs"


def test_file_seed_depends_on_the_base_seed_and_the_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_spectra(data)
    seen = {}
    real_fit = fit_api.run_fit_consistent

    def spy(x, y, peaks_in, config, *args, **kwargs):
        seen.setdefault(config["perf_seed"], []).append(kwargs.get("rng_seed"))
        return real_fit(x, y, peaks_in, config, *args, **kwargs)

    monkeypatch.setattr(fit_api, "run_fit_consistent", spy)
    for base_seed in (1, 2):
        cfg = _config(tmp_path / f"out{base_seed}", base_seed)
        cfg["solver_restarts"] = 1
        runner.run_batch([str(data / "*.csv")], cfg, compute_uncertainty=False)

    (a1, b1), (a2, b2) = seen[1], seen[2]
    assert a1 != b1  # each file has its own seed
    assert a1 != a2 and b1 != b2  # and the base seed changes it
