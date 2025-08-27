import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io
from core.peaks import Peak
from ui.app import pseudo_voigt  # noqa: E402


def _headers(n):
    x = np.linspace(-1, 1, 5)
    y = np.zeros_like(x)
    peaks = []
    for i in range(n):
        p = Peak(float(i), 1.0, 1.0, 0.5)
        y += pseudo_voigt(x, 1.0, p.center, p.fwhm, p.eta)
        peaks.append(p)
    table = data_io.build_trace_table(x, y, np.zeros_like(x), peaks)
    return table.splitlines()[0].split(',')


def test_trace_schema_v27():
    for n in (0, 1, 2):
        hdr = _headers(n)
        expect = ["x", "y_raw", "baseline", "y_target_add", "y_fit_add"]
        expect += [f"peak{i+1}" for i in range(n)]
        expect += ["y_target_sub", "y_fit_sub"]
        expect += [f"peak{i+1}_sub" for i in range(n)]
        assert hdr == expect
