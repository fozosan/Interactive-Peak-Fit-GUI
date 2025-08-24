import sys
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import data_io
from core.peaks import Peak
from core.models import pv_sum

def test_trace_table_includes_baseline_added_and_subtracted():
    x = np.linspace(0, 4, 5)
    baseline = 0.5 * x
    peak = Peak(2.0, 1.0, 1.0, 0.5)
    y = pv_sum(x, [peak]) + baseline

    csv_str = data_io.build_trace_table(x, y, baseline, [peak])
    lines = list(csv.reader(csv_str.strip().splitlines()))
    header = lines[0]
    assert header[:5] == ["x", "y_raw", "baseline", "y_corr", "y_fit"]
    assert header[5] == "peak1"

    first = list(map(float, lines[1]))
    x0, y_raw0, base0, corr0, fit0, comp0 = first
    assert np.isclose(y_raw0 - base0, corr0)
    assert np.isclose(fit0, comp0 + base0)
