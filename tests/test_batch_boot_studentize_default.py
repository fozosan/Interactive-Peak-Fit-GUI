"""The batch bootstrap studentizes its residuals by default (game plan 1.7, PR E's follow-up).

``boot_studentize`` resamples r_i / sqrt(1 - h_ii): the residuals with their
leverage shrinkage undone, h the hat matrix of the weighted Jacobian with the
joint baseline's columns. The few points that carry a narrow line's height have
leverage near 1, so without it the intervals come out too narrow. Measured
through ``run_batch`` (joint baseline, estimated noise, B = 500, PR F), paired
on 400 replicates, studentizing gains 0.6 to 2.2 points in every cell and loses
none: strong MoS2 centre / height / FWHM 91.1 / 89.2 / 92.0 off, 91.9 / 90.1 /
93.4 on; weak MoS2 reads 94.4 / 94.0 / 94.5 on and Si 93.0 / 95.2 / 94.2. The
core default stays off, so the GUI, whose checkbox shows it, is unchanged.
"""

from batch import runner
from tests.synthetic_raman import SI_BACKGROUND, SI_CURVE_SCALE, SI_LINES, SI_TEMPLATE, SI_WINDOW, curved_shape, mos2_spectrum, write_spectrum


def _studentize_seen(tmp_path, monkeypatch, extra):
    seen = []
    real = runner.bootstrap_ci

    def spy(*args, **kwargs):
        res = real(*args, **kwargs)
        seen.append((kwargs["fit_ctx"].get("boot_studentize"), res.diagnostics.get("boot_studentize")))
        return res

    monkeypatch.setattr(runner, "bootstrap_ci", spy)
    folder = tmp_path / str(len(list(tmp_path.iterdir())))
    folder.mkdir()
    spec = folder / "si.csv"
    write_spectrum(spec, *mos2_spectrum(1, SI_LINES, background=lambda x: SI_BACKGROUND + curved_shape(x, SI_WINDOW, SI_CURVE_SCALE)))
    cfg = {
        "peaks": [dict(p) for p in SI_TEMPLATE],
        "solver": "modern_vp",
        "source": "template",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": str(folder / "out"),
        "fit_xmin": SI_WINDOW[0],
        "fit_xmax": SI_WINDOW[1],
        "bootstrap_n": 8,
        "unc_boot_solver": "modern_vp",
        "perf_seed_all": True,
        "perf_seed": 3,
        **extra,
    }
    runner.run_batch([str(spec)], cfg, compute_uncertainty=True, unc_method="bootstrap")
    return seen


def test_batch_bootstrap_studentizes_by_default(tmp_path, monkeypatch):
    assert _studentize_seen(tmp_path, monkeypatch, {}) == [(True, True)]
    # and the setting still turns it off
    assert _studentize_seen(tmp_path, monkeypatch, {"boot_studentize": False}) == [(False, False)]
