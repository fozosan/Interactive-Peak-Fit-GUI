"""The batch bootstrap starts every refit at the fit and reports the fit.

Its start jitter defaulted to 2 % of each parameter's absolute value: 7.7 cm-1
on a centre at 383 cm-1, three times the line width. The refits then landed in
other minima, and on MoS2-like spectra the bootstrap intervals came out 137 to
1595 times the true width. The exported value was the mean of those refits,
1.2 to 6.4 cm-1 away from the fit.

Seeded, three replicates of the measured MoS2 noise and 30 refits each: enough
to tell an interval the size of the asymptotic one from one a hundred times
wider, in a few seconds. The two do not agree exactly: the app's ALS baseline
sits about a count low, which the asymptotic counts as noise (RSS / dof) and
the bootstrap's centred residuals do not, so the bootstrap comes out about 0.7
times the asymptotic. Hence the band of a factor 3 either way.
"""

import csv

import pytest

from batch import runner
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum


def _config(out_dir, window):
    cfg = {
        "peaks": [dict(p) for p in TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(out_dir),
        "output_base": "batch",
        "bootstrap_n": 30,
        "unc_boot_solver": "modern_vp",
        "perf_seed_all": True,
        "perf_seed": 7,
        "perf_unc_workers": 1,
    }
    if window:
        cfg["fit_xmin"], cfg["fit_xmax"] = WINDOW
    return cfg


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("window", [True, False], ids=["window_340_440", "full_range"])
def test_bootstrap_centre_interval_is_the_size_of_the_asymptotic_one(tmp_path, window):
    for rep in range(3):
        folder = tmp_path / f"rep{rep}"
        folder.mkdir()
        spectrum = folder / "spec.csv"
        write_spectrum(spectrum, *mos2_spectrum(100 + rep))
        se = {}
        for method in ("asymptotic", "bootstrap"):
            out = folder / method
            runner.run_batch(
                [str(spectrum)], _config(out, window), compute_uncertainty=True, unc_method=method
            )
            status = _rows(out / "batch_files.csv")[0]
            assert status["unc_status"] == "ok", status["unc_message"]
            fitted = {int(r["peak"]): float(r["center"]) for r in _rows(out / "spec_fit.csv")}
            centre = {
                int(r["peak"]): (float(r["value"]), float(r["stderr"]))
                for r in _rows(out / "spec_uncertainty.csv")
                if r["param"] == "center"
            }
            for peak, (value, _sd) in centre.items():
                # The exported value is the fit, not the mean of the refits.
                assert value == pytest.approx(fitted[peak], rel=1e-12), (method, peak)
            se[method] = {peak: sd for peak, (_v, sd) in centre.items()}
        for peak in (1, 2):
            ratio = se["bootstrap"][peak] / se["asymptotic"][peak]
            assert 1 / 3 < ratio < 3.0, f"replicate {rep}, peak {peak}: bootstrap/asymptotic SE = {ratio:.3g}"
