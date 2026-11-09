"""Fast seeded regression of the Bayesian through ``run_batch``: posterior widths of the right size.

The coverage check of game plan 1.7 runs hundreds of replicates per case. This
is its quick version: two spectra of each case (strong and weak MoS2, Si) with
the measured count noise over the curved background measured on the 5 s MoS2
spectrum, through ``run_batch`` with its defaults (per-spectrum noise model,
joint baseline). The Bayesian posterior sd must match the asymptotic SE of the
same spectrum: with flat priors and this much data the two agree to a few per
cent (measured 0.95 to 1.10 on one replicate of each case, PR F probe
proto_tau.py). Before this PR the walkers started up to 1e9 counts out (the
solver's 1e12 height bound), and PR E measured height widths thousands of times
the true sd.
"""

import csv

import numpy as np
import pytest

from batch import runner
from tests.synthetic_raman import (
    BACKGROUND, MOS2_LINES, MOS2_WEAK_LINES, SI_BACKGROUND, SI_CURVE_SCALE, SI_LINES, SI_TEMPLATE, SI_WINDOW,
    TEMPLATE, WEAK_BACKGROUND, WINDOW, curved_shape, mos2_spectrum, write_spectrum,
)

pytest.importorskip("emcee")

CASES = {
    "mos2_strong": (MOS2_LINES, WINDOW, TEMPLATE, lambda x: BACKGROUND + curved_shape(x)),
    "mos2_weak": (MOS2_WEAK_LINES, WINDOW, TEMPLATE, lambda x: WEAK_BACKGROUND + curved_shape(x)),
    "si": (SI_LINES, SI_WINDOW, SI_TEMPLATE, lambda x: SI_BACKGROUND + curved_shape(x, SI_WINDOW, SI_CURVE_SCALE)),
}


def _batch(folder, case, seed, method):
    lines, window, template, background = CASES[case]
    folder.mkdir()
    spec = folder / "spec.csv"
    write_spectrum(spec, *mos2_spectrum(seed, lines, background=background))
    cfg = {
        "peaks": [dict(p) for p in template],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(folder / "out"),
        "fit_xmin": window[0],
        "fit_xmax": window[1],
        "perf_seed_all": True,
        "perf_seed": 100 + seed,
        "bayes_walkers": 0,
        "bayes_burn": 300,
        "bayes_steps": 1500,
        "bayes_thin": 1,
    }
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method=method)
    with open(folder / "out" / "spec_uncertainty.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    sd = np.full((len(lines), 3), np.nan)
    for r in rows:
        if r["param"] in ("center", "height", "fwhm"):
            sd[int(r["peak"]) - 1, ("center", "height", "fwhm").index(r["param"])] = float(r["stderr"])
    return sd


@pytest.mark.parametrize("case", sorted(CASES))
def test_bayesian_widths_match_the_asymptotic(tmp_path, case):
    for seed in (0, 1):
        se = _batch(tmp_path / f"asy{seed}", case, seed, "asymptotic")
        sd = _batch(tmp_path / f"bay{seed}", case, seed, "bayesian")
        ratio = sd / se
        assert np.all((ratio > 0.8) & (ratio < 1.25)), (case, seed, ratio)
