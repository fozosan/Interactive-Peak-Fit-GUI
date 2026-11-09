"""Fast seeded regression of the joint baseline: unbiased fits, intervals of the true size.

The coverage check of game plan 1.6 runs 200 replicates per case. This is its
quick version: a few replicates of the measured count noise over a realistic
curved background (measured on the 5 s MoS2 spectrum, ``curved_shape``),
through ``run_batch`` with its defaults: the per-spectrum noise model, which it
estimates itself, and the baseline fitted with the peaks. The estimates are
compared with the truth, and the asymptotic SEs with the true sd of the joint
estimator: the Fisher information at the truth, with the true noise and the
peak and cubic-baseline columns.

Holding the default ALS baseline fixed instead (the review's finding 4: it sits
about two noise sd low and bends into strong lines) moves the Si FWHM by about
19 true sd and the weak MoS2 FWHM by tens of sd.

The bounds come from 40 seeds of each case (PR E probe fasttest_dist.py): one
spectrum's noise estimate moves its SEs by -50 to +60 % (1st to 99th
percentile), and the Si SEs average 0.89 of the true sd, the weak MoS2 SEs 1.08.
"""

import csv

import numpy as np
import pytest

from batch import runner
from core.models import pseudo_voigt
from tests.synthetic_raman import (
    BIAS, MOS2_WEAK_LINES, NOISE_FLOOR, NOISE_GAIN, SI_BACKGROUND, SI_CURVE_SCALE, SI_LINES, SI_TEMPLATE,
    SI_WINDOW, TEMPLATE, WEAK_BACKGROUND, WINDOW, curved_shape, mos2_spectrum, mos2_x, write_spectrum,
)

CASES = {
    "si": (SI_LINES, SI_WINDOW, SI_TEMPLATE, lambda x: SI_BACKGROUND + curved_shape(x, SI_WINDOW, SI_CURVE_SCALE)),
    "mos2_weak": (MOS2_WEAK_LINES, WINDOW, TEMPLATE, lambda x: WEAK_BACKGROUND + curved_shape(x)),
}


def _true_sd(lines, window, background):
    """sd of the joint estimator at the truth: (J^T V^-1 J)^-1 with the peaks' and a cubic's columns."""
    x = mos2_x()
    x = x[(x >= window[0]) & (x <= window[1])]
    mu = background(x).copy()
    cols = []
    for c, h, w, e in lines:
        f0 = pseudo_voigt(x, h, c, w, e)
        mu += f0
        for dc, dh, dw in ((1e-5, 0.0, 0.0), (0.0, 1e-5 * h, 0.0), (0.0, 0.0, 1e-5 * w)):
            cols.append((pseudo_voigt(x, h + dh, c + dc, w + dw, e) - f0) / (dc + dh + dw))
    u = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    cols += list(np.polynomial.legendre.legvander(u, 3).T)
    J = np.column_stack(cols) / np.sqrt(NOISE_FLOOR + NOISE_GAIN * (mu - BIAS))[:, None]
    return np.sqrt(np.diag(np.linalg.inv(J.T @ J)))[: 3 * len(lines)].reshape(-1, 3)


def _batch(tmp_path, seed, case, baseline_fit):
    lines, window, template, background = CASES[case]
    folder = tmp_path / f"{case}_{baseline_fit}_{seed}"
    folder.mkdir()
    spec = folder / "spec.csv"
    write_spectrum(spec, *mos2_spectrum(seed, lines, background=background))
    cfg = {
        "peaks": [dict(p) for p in template],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(folder / "out"),
        "output_base": "batch",
        "fit_xmin": window[0],
        "fit_xmax": window[1],
    }
    if baseline_fit is not None:
        cfg["baseline_fit"] = baseline_fit
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="asymptotic")
    with open(folder / "out" / "spec_uncertainty.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    est = np.full((len(lines), 3), np.nan)
    se = np.full((len(lines), 3), np.nan)
    for r in rows:
        if r["param"] in ("center", "height", "fwhm"):
            k = ("center", "height", "fwhm").index(r["param"])
            est[int(r["peak"]) - 1, k] = float(r["value"])
            se[int(r["peak"]) - 1, k] = float(r["stderr"])
    return est, se


@pytest.mark.parametrize("case,seeds", [("si", range(800, 806)), ("mos2_weak", range(810, 815))])
def test_the_joint_fit_is_unbiased_and_its_intervals_the_true_size(tmp_path, case, seeds):
    lines = CASES[case][0]
    truth = np.array([[c, h, w] for c, h, w, _e in lines])
    sd = _true_sd(*CASES[case][:2], CASES[case][3])
    z, ratio = [], []
    for s in seeds:
        est, se = _batch(tmp_path, s, case, None)  # the batch default: joint
        z.append((est - truth) / sd)
        ratio.append(se / sd)
    z, ratio = np.array(z), np.array(ratio)
    # per value: within the scatter of one spectrum's noise estimate
    assert np.all((ratio > 0.45) & (ratio < 2.0)), ratio
    # on average the intervals are the right size and the estimates on the truth
    assert np.all((ratio.mean(axis=0) > 0.75) & (ratio.mean(axis=0) < 1.3)), ratio.mean(axis=0)
    assert np.all(np.abs(z.mean(axis=0)) < 1.5), z.mean(axis=0)
    assert np.all(np.abs(z) < 4.0), z


def test_joint_removes_the_width_bias_of_a_fixed_als_baseline(tmp_path):
    for case, seed, sign in (("si", 800, -1.0), ("mos2_weak", 810, 1.0)):
        lines = CASES[case][0]
        truth = np.array([[c, h, w] for c, h, w, _e in lines])
        sd = _true_sd(*CASES[case][:2], CASES[case][3])
        fixed = (_batch(tmp_path, seed, case, "fixed")[0] - truth) / sd
        joint = (_batch(tmp_path, seed, case, "joint")[0] - truth) / sd
        assert sign * fixed[:, 2].mean() > 5.0, (case, fixed)
        assert np.all(np.abs(joint[:, 2]) < 3.0), (case, joint)
