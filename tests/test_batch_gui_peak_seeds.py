"""Run Batch with Current or Template seeds passes the GUI's own peak dicts (game plan Phase 0.5).

``ui.app.Peak`` carries ``fwhm_cap_enabled`` and ``fwhm_cap_value``, which
``core.peaks.Peak`` does not have, so every such batch raised TypeError before
it opened a file. The caps reach the solvers as ``width_caps``: the solver
options win when they carry a cap, and the peaks' own caps are used otherwise.
"""
import csv

import numpy as np
import pytest

from batch import runner
from core import models, peaks
from ui.app import Peak as GuiPeak


def _fit_one(tmp_path, records, solver_options):
    x = np.linspace(0.0, 100.0, 201)
    y = models.pv_sum(x, [peaks.Peak(50.0, 100.0, 8.0, 0.5)]) + 10.0  # true FWHM 8
    np.savetxt(tmp_path / "a.csv", np.column_stack([x, y]), delimiter=",")
    cfg = {
        "peaks": records,
        "source": "template",
        "solver": "modern_vp",
        "modern_vp": solver_options,
        "mode": "add",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "baseline_uses_fit_range": True,
        "fit_xmin": 20.0,
        "fit_xmax": 80.0,
        "output_dir": str(tmp_path / "out"),
        "output_base": "batch",
    }

    ok, n = runner.run_batch([str(tmp_path / "a.csv")], cfg, compute_uncertainty=False)

    assert (ok, n) == (1, 1)
    with open(tmp_path / "out" / "batch_fit.csv", newline="", encoding="utf-8") as fh:
        (row,) = list(csv.DictReader(fh))
    return float(row["fwhm"])


@pytest.mark.parametrize(
    "solver_caps, binding_cap",
    [
        ("absent", 5.0),   # the GUI peak's own cap applies
        ([None], 5.0),     # options with no cap in them count as absent
        ([7.0], 7.0),      # a cap in the solver options wins
    ],
)
def test_gui_peak_dicts_fit_and_keep_their_caps(tmp_path, solver_caps, binding_cap):
    seed = GuiPeak(49.0, 80.0, 4.0, 0.5, fwhm_cap_enabled=True, fwhm_cap_value=5.0)
    options = {"centers_in_window": True}
    if solver_caps != "absent":
        options["width_caps"] = solver_caps

    fwhm = _fit_one(tmp_path, [dict(seed.__dict__)], options)

    assert fwhm == pytest.approx(binding_cap, abs=1e-3)


def test_an_uncapped_gui_peak_fits_its_true_width(tmp_path):
    seed = GuiPeak(49.0, 80.0, 4.0, 0.5)  # cap off, the GUI's default
    record = dict(seed.__dict__, label="A1g")  # and a key no Peak has

    fwhm = _fit_one(tmp_path, [record], {"centers_in_window": True})

    assert fwhm == pytest.approx(8.0, abs=0.3)


def test_peak_records_adapter():
    records = [
        dict(GuiPeak(1.0, 2.0, 3.0, 0.4, lock_width=True, fwhm_cap_enabled=True).__dict__),
        dict(GuiPeak(5.0, 6.0, 7.0, 0.6, lock_center=True, fwhm_cap_enabled=True,
                     fwhm_cap_value=9.0).__dict__),
        {"center": 8.0, "height": 1.0, "fwhm": 2.0, "eta": 0.5},
    ]

    assert peaks.deserialize(records) == [
        peaks.Peak(1.0, 2.0, 3.0, 0.4, lock_width=True),
        peaks.Peak(5.0, 6.0, 7.0, 0.6, lock_center=True),
        peaks.Peak(8.0, 1.0, 2.0, 0.5),
    ]
    # A locked width has no cap; that is the rule the GUI uses for its own peaks.
    assert peaks.width_caps(records) == [None, 9.0, None]
