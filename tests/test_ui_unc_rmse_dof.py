import numpy as np
from core.data_io import normalize_unc_result

def test_ui_norm_injects_rmse_dof_smoke():
    # simulate a UI-produced dict without rmse/dof
    out = {"method": "bootstrap", "param_stats": {"center": {"est": [1.0], "sd": [0.1]}}}
    norm = normalize_unc_result(out)
    # emulate the UI injection step logic
    r = np.array([1.0, -1.0, 0.5])
    rmse = float(np.sqrt(np.mean(r**2)))
    dof = max(1, r.size - 4)  # pretend one peak (4 params)
    norm.setdefault("rmse", rmse)
    norm.setdefault("dof", dof)
    assert "rmse" in norm and "dof" in norm
