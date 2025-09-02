import pandas as pd
from core import data_io


def test_batch_summary_no_nan(tmp_path):
    rows = [
        {"file":"a.csv","peak":1,"param":"center","value": float("nan"), "stderr":float("nan"), "method":"Asymptotic","rmse": float("nan"), "dof": 0},
        {"file":"a.csv","peak":1,"param":"height","value": 3.0, "stderr":0.0, "p2_5": float("nan"), "p97_5": float("nan"), "method":"Asymptotic","rmse": 0.1, "dof": 0},
    ]
    long_path, legacy_path = data_io.write_batch_uncertainty_long(tmp_path, rows)
    df = pd.read_csv(long_path)
    assert not df[["value","stderr","p2_5","p97_5","rmse","dof"]].isna().any().any()
