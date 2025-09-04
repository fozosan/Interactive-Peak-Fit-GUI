import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core import data_io  # noqa: E402


def test_unc_text_includes_label_and_pm(tmp_path):
    res_dict = {"type": "bootstrap", "params": {"p1": {"est": 1.0, "sd": 0.2}}}
    path = tmp_path / "unc.txt"
    data_io.write_uncertainty_txt(path, res_dict)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "Bootstrap" in lines[0]
    assert "±" in lines[1]

