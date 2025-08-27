import numpy as np
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io
from core.peaks import Peak
from ui.app import pseudo_voigt  # noqa: E402


def test_trace_schema_add_then_sub():
    x = np.linspace(-2, 2, 5)
    peak = Peak(0.0, 1.0, 1.0, 0.5)
    baseline = np.linspace(0.5, 1.0, x.size)
    comp = pseudo_voigt(x, peak.height, peak.center, peak.fwhm, peak.eta)
    y = baseline + comp
    table = data_io.build_trace_table(x, y, baseline, [peak])
    lines = table.splitlines()
    header = lines[0].split(',')
    expect = ["x", "y_raw", "baseline", "y_target_add", "y_fit_add", "peak1", "y_target_sub", "y_fit_sub", "peak1_sub"]
    assert header == expect
    arr = np.genfromtxt(lines[1:], delimiter=',')
    b = header.index("baseline")
    p_add = header.index("peak1")
    p_sub = header.index("peak1_sub")
    assert np.allclose(arr[:, p_add], arr[:, b] + arr[:, p_sub])
