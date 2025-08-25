import numpy as np
import tempfile, os
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.peaks import Peak
from core.models import pv_sum
from batch import runner


def test_batch_smoke():
    # Build a tiny temp folder of 2 synthetic spectra and run batch runner
    with tempfile.TemporaryDirectory() as td:
        xs = np.linspace(0, 50, 501)
        for i, shift in enumerate([0.0, 0.5], start=1):
            peaks = [Peak(20 + shift, 5, 5, 0.5)]
            ys = pv_sum(xs, peaks)
            data = np.column_stack([xs, ys])
            np.savetxt(os.path.join(td, f"s{i}.csv"), data, delimiter=",", header="x,y", comments="")

        out_csv = os.path.join(td, "out.csv")
        cfg = {
            "peaks": [{"center": 20, "height": 5, "fwhm": 5, "eta": 0.5}],
            "solver": "classic",
            "mode": "add",
            "baseline": {"lam": 1e5, "p": 0.001},
            "peak_output": out_csv,
        }
        runner.run([os.path.join(td, "*.csv")], cfg)
        assert os.path.exists(out_csv)
