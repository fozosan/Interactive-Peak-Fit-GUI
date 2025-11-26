import sys, pathlib, numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from batch import runner
from core.peaks import Peak
from tests.conftest import bayes_knobs, bootstrap_cfg, ensure_unc_common
from ui.app import pseudo_voigt


def test_batch_toggle_respected(tmp_path):
    x = np.linspace(-5,5,101)
    pk = Peak(0.0,1.0,1.0,0.5)
    for i in range(2):
        y = pseudo_voigt(x, pk.height, pk.center, pk.fwhm, pk.eta)
        np.savetxt(tmp_path / f"f{i}.csv", np.column_stack([x,y]), delimiter=",")

    cfg = {
        "peaks":[pk.__dict__],
        "solver":"classic",
        "mode":"add",
        "baseline":{"lam":1e5,"p":0.001,"niter":10,"thresh":0.0},
        "save_traces":False,
        "source":"template",
        "reheight":False,
        "auto_max":5,
        "classic":{},
        "baseline_uses_fit_range":True,
        "perf_numba":False,"perf_gpu":False,"perf_cache_baseline":True,"perf_seed_all":False,"perf_max_workers":1,
        "output_dir":str(tmp_path),
        "output_base":"batch",
    }
    cfg.update(ensure_unc_common({}))
    cfg.update(bootstrap_cfg(n=80))
    cfg.update(bayes_knobs())

    # OFF: no per-file unc files, no batch uncertainty
    ok, total = runner.run_batch([str(tmp_path / "f?.csv")], cfg, compute_uncertainty=False, unc_method="Asymptotic")
    assert ok==2 and total==2
    assert not (tmp_path / "f0_uncertainty.csv").exists()
    assert not (tmp_path / "batch_uncertainty.csv").exists()

    # ON: files must exist
    ok, total = runner.run_batch([str(tmp_path / "f?.csv")], cfg, compute_uncertainty=True, unc_method="Asymptotic")
    assert ok==2 and total==2
    assert (tmp_path / "f0_uncertainty.csv").exists()
    assert (tmp_path / "batch_uncertainty.csv").exists()
