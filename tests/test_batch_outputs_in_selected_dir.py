"""Batch outputs should be written to the selected directory only."""

import numpy as np

from batch import runner
from core import models, peaks


def test_batch_outputs_in_selected_dir(tmp_path, no_blank_lines):
    data_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()

    x = np.linspace(-5, 5, 101)
    pk = peaks.Peak(0.0, 1.0, 1.0, 0.5)
    y = models.pv_sum(x, [pk])
    arr = np.column_stack([x, y])
    for i in range(2):
        np.savetxt(data_dir / f"s{i}.csv", arr, delimiter=",")

    cfg = {
        "peaks": [pk.__dict__],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": True,
        "source": "template",
        "output_dir": str(out_dir),
        "output_base": "batch",
    }

    runner.run([str(data_dir / "*.csv")], cfg)

    for stem in ("s0", "s1"):
        for suf in ("_fit.csv", "_trace.csv", "_uncertainty.csv", "_uncertainty.txt"):
            out_path = out_dir / f"{stem}{suf}"
            assert out_path.exists()
            assert not (data_dir / f"{stem}{suf}").exists()
            if suf.endswith(".csv"):
                assert no_blank_lines(out_path)

    for summary in ("batch_fit.csv", "batch_uncertainty.csv"):
        path = out_dir / summary
        assert path.exists()
        assert no_blank_lines(path)

