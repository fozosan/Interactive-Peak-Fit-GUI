import sys, pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io  # noqa: E402


def test_export_no_blank_lines(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "out.csv"
    data_io.write_dataframe(df, path)
    text = path.read_text()
    assert "\n\n" not in text
    assert text.endswith("\n")
