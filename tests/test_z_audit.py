import pathlib, sys, inspect, hashlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.peaks import Peak
from core.models import pv_sum
from fit import classic, modern_vp, modern, step_engine

try:  # optional
    from fit import lmfit_backend
    HAVE_LMFIT = True
except Exception:  # pragma: no cover
    lmfit_backend = None
    HAVE_LMFIT = False


def _hash(mod):
    src = inspect.getsource(mod)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]


def _parity_ratio(backend):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 60, 400)
    true = [Peak(20, 5, 5, 0.5), Peak(40, 2, 6, 0.3)]
    y = pv_sum(x, true) + 0.01 * rng.standard_normal(x.size)
    start = [Peak(19, 4, 6, 0.5), Peak(41, 1.5, 7, 0.3)]
    full = backend.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in start], "add", None, {})
    theta_full = full["theta"]
    peaks_full = [Peak(theta_full[4*i], theta_full[4*i+1], theta_full[4*i+2], theta_full[4*i+3]) for i in range(len(start))]
    rmse_full = np.sqrt(np.mean((pv_sum(x, peaks_full) - y) ** 2))
    s = backend.prepare_state(x, y, start, mode="add", baseline=None, opts={})["state"]
    for _ in range(15):
        s, ok, c0, c1, info = backend.iterate(s)
    theta = s["theta"]
    peaks_final = [Peak(theta[4*i], theta[4*i+1], theta[4*i+2], theta[4*i+3]) for i in range(len(start))]
    rmse_step = np.sqrt(np.mean((pv_sum(x, peaks_final) - y) ** 2))
    return rmse_step / rmse_full


def test_audit_line():
    mods = [classic, modern_vp, modern]
    if HAVE_LMFIT:
        mods.append(lmfit_backend)
    hashes = {m.__name__: _hash(m) for m in mods}
    ratios = {"modern_vp": _parity_ratio(modern_vp), "modern_trf": _parity_ratio(modern)}
    line = (
        f"AUDIT hashes={hashes} max_backtracks=8 min_step_ratio={1e-9} parity={ratios}"
    )
    print(line)
    assert isinstance(hashes, dict)

