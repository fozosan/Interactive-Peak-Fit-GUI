"""Fast seeded regression of the weighting: the batch's intervals are the right size.

The coverage check of game plan 1.5 runs 200 replicates per case. This is its
quick version: a few replicates of the measured CCD noise, var = 0.724 + 0.0451 *
(mu - 502.5), through ``run_batch`` with its default per-spectrum noise model and
the true flat baseline, so only the noise model differs from the truth. The
batch estimates s0 and g itself (never handed the truth). Its asymptotic SEs
are compared with the true sd of the weighted estimator, from the Fisher
information at the truth with the true noise.

Before the noise model the strong lines' height SE came out about 0.55 of the
true sd (review: coverage 61 to 76 %); one sigma for every point cannot see that
a 90-count peak top has 6.6 times the floor variance.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core.models import pseudo_voigt
from tests.synthetic_raman import (
    BACKGROUND, BIAS, MOS2_LINES, NOISE_FLOOR, NOISE_GAIN, TEMPLATE, WINDOW, mos2_spectrum, mos2_x, write_spectrum,
)

SI_LINE = ((520.02, 2142.5, 6.62, 0.5),)
SI_WINDOW = (490.0, 550.0)
SI_TEMPLATE = [{"center": 520.5, "height": 2000.0, "fwhm": 4.0, "eta": 0.5}]


def _true_sd(lines, window):
    """sd of the weighted estimator at the truth: (J^T V^-1 J)^-1 with the true noise."""
    x = mos2_x()
    x = x[(x >= window[0]) & (x <= window[1])]
    cols, mu = [], np.full_like(x, BACKGROUND)
    for c, h, w, e in lines:
        u = (x - c) / w
        lor = 1.0 / (1.0 + 4.0 * u * u)
        gau = np.exp(-4.0 * np.log(2.0) * u * u)
        d = e * (-8.0 * u) * lor * lor + (1.0 - e) * gau * (-8.0 * np.log(2.0) * u)
        cols += [h * d * (-1.0 / w), e * lor + (1.0 - e) * gau, h * d * (-u / w)]
        mu = mu + pseudo_voigt(x, h, c, w, e)
    J = np.column_stack(cols) / np.sqrt(NOISE_FLOOR + NOISE_GAIN * (mu - BIAS))[:, None]
    return np.sqrt(np.diag(np.linalg.inv(J.T @ J))).reshape(-1, 3)


def _batch_se(tmp_path, seed, lines, window, template, noise_model, monkeypatch):
    monkeypatch.setattr(runner.signals, "als_baseline", lambda y, **k: np.full_like(np.asarray(y, float), BACKGROUND))
    folder = tmp_path / f"{noise_model}_{seed}"
    folder.mkdir()
    spec = folder / "spec.csv"
    write_spectrum(spec, *mos2_spectrum(seed, lines))
    cfg = {
        "peaks": [dict(p) for p in template],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(folder / "out"),
        "output_base": "batch",
        "fit_xmin": window[0],
        "fit_xmax": window[1],
        "noise_model": noise_model,
    }
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="asymptotic")
    with open(folder / "out" / "spec_uncertainty.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    se = np.zeros((len(lines), 3))
    for r in rows:
        if r["param"] in ("center", "height", "fwhm"):
            se[int(r["peak"]) - 1, ("center", "height", "fwhm").index(r["param"])] = float(r["stderr"])
    return se


@pytest.mark.parametrize(
    "lines,window,template,seeds",
    [(MOS2_LINES, WINDOW, TEMPLATE, range(600, 605)), (SI_LINE, SI_WINDOW, SI_TEMPLATE, range(700, 703))],
    ids=["mos2_strong", "si"],
)
def test_weighted_intervals_have_the_true_size(tmp_path, monkeypatch, lines, window, template, seeds):
    sd = _true_sd(lines, window)
    ratios = np.array([_batch_se(tmp_path, s, lines, window, template, "per_spectrum", monkeypatch) / sd for s in seeds])
    # per value: within the scatter of one spectrum's g estimate
    assert np.all((ratios > 0.6) & (ratios < 1.6)), ratios
    # on average over the replicates, centre, height and FWHM are the right size
    mean = ratios.mean(axis=(0, 1))
    assert np.all((mean > 0.85) & (mean < 1.2)), mean
    # one sigma for every point, as before: the height interval is far too narrow
    plain = np.array([_batch_se(tmp_path, s, lines, window, template, "none", monkeypatch) / sd for s in list(seeds)[:2]])
    assert plain[:, :, 1].mean() < 0.75, plain
