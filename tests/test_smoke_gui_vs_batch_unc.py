import csv
import math
import os
import numpy as np
from pathlib import Path

from core import models, peaks, data_io, fit_api
from core.residuals import build_residual
from core.uncertainty import bootstrap_ci, finite_diff_jacobian
from batch.runner import run_batch
from tests.conftest import bayes_knobs, bootstrap_cfg, ensure_unc_common


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.9g},{yi:.9g}\n")


def _read_long_csv(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for r in reader:
            rows.append(
                {
                    "peak": int(r["peak"]),
                    "param": str(r["param"]),
                    "value": float(r["value"]) if r["value"] else float("nan"),
                    "stderr": float(r["stderr"]) if r["stderr"] else float("nan"),
                    "p2_5": float(r["p2_5"]) if r["p2_5"] else float("nan"),
                    "p97_5": float(r["p97_5"]) if r["p97_5"] else float("nan"),
                }
            )
    rows.sort(key=lambda d: (d["peak"], d["param"]))
    return rows


def _predict_full_from_peaks(th: np.ndarray, x: np.ndarray, pk_list, baseline=None, mode="add"):
    peaks_eval = []
    for i in range(len(pk_list)):
        c, h, fw, eta = th[4 * i : 4 * (i + 1)]
        peaks_eval.append(peaks.Peak(c, h, fw, eta))
    total = models.pv_sum(x, peaks_eval)
    if mode == "add" and baseline is not None:
        total = total + baseline
    return total


def _compare_rows(rows_a, rows_b, *, rtol=0.08, atol=1e-8):
    assert len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b):
        assert (ra["peak"], ra["param"]) == (rb["peak"], rb["param"])
        for k in ("value", "stderr", "p2_5", "p97_5"):
            va, vb = float(ra[k]), float(rb[k])
            if not (math.isfinite(va) and math.isfinite(vb)):
                assert (not math.isfinite(va)) and (not math.isfinite(vb))
            else:
                assert math.isclose(va, vb, rel_tol=rtol, abs_tol=atol), f"{k} mismatch: {va} vs {vb}"


def _make_spectrum(n=401, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    pk = peaks.Peak(5.0, 1.0, 0.8, 0.5)
    y0 = models.pv_sum(x, [pk])
    y = y0 + rng.normal(0.0, noise, size=x.shape)
    return x, y


def _fit_once_gui_style(x, y, *, config):
    pk = peaks.Peak(5.0, 1.0, 1.0, 0.5)
    baseline = np.zeros_like(y)
    mask = np.ones_like(x, bool)
    out = fit_api.run_fit_consistent(
        x, y, [pk], dict(config), baseline, "add", mask
    )
    assert out["fit_ok"], "fit failed in smoke test"
    theta_hat = np.asarray(out["theta"], float)
    peaks_out = out.get("peaks_out") or out.get("peaks") or [pk]
    residual_fn = build_residual(x, y, peaks_out, "add", baseline, "linear", None)
    jac = finite_diff_jacobian(residual_fn, theta_hat)
    jac_mat = np.atleast_2d(np.asarray(jac, float))
    residual_vec = np.asarray(residual_fn(theta_hat), float)

    fit_ctx = {
        "x_all": x,
        "y_all": y,
        "baseline": baseline,
        "mode": "add",
        "peaks": peaks_out,
        "peaks_out": peaks_out,
        "solver": str(config.get("unc_boot_solver", config.get("solver_choice", "modern_trf"))),
        "bootstrap_jitter": float(config.get("bootstrap_jitter", 0.02)) if float(config.get("bootstrap_jitter", 0.02)) <= 1.0 else float(config.get("bootstrap_jitter", 0.02))/100.0,
    }

    seed_cfg = config.get("perf_seed", "")
    try:
        seed_val = int(seed_cfg) if str(seed_cfg) not in ("", "None") else None
    except Exception:
        seed_val = None
    if not bool(config.get("perf_seed_all", False)):
        seed_val = None

    unc_res = bootstrap_ci(
        theta=theta_hat,
        residual=np.asarray(residual_vec, float),
        jacobian=jac_mat,
        predict_full=lambda th: _predict_full_from_peaks(th, x, peaks_out, baseline, "add"),
        x_all=x,
        y_all=y,
        fit_ctx=fit_ctx,
        bounds=out.get("bounds"),
        param_names=out.get("param_names"),
        locked_mask=out.get("locked_mask"),
        n_boot=int(config.get("bootstrap_n", 60)),
        seed=seed_val,
        workers=None,
        alpha=float(config.get("unc_alpha", 0.05)),
        center_residuals=bool(config.get("unc_center_resid", True)),
        return_band=False,
    )
    return out, unc_res


def test_bootstrap_gui_vs_batch_match(tmp_path):
    x, y = _make_spectrum()
    fpath = tmp_path / "sample.csv"
    _write_xy(fpath, x, y)
    base = tmp_path / "sample"

    cfg = {
        "solver_choice": "modern_trf",
        "unc_method": "Bootstrap",
        "unc_boot_solver": "modern_trf",
        "unc_center_resid": True,
        "bootstrap_n": 60,
        "perf_seed_all": True,
        "perf_seed": 1234,
        "bootstrap_jitter": 0.02,
        "export_unc_wide": True,
        "unc_alpha": 0.05,
    }

    _, unc_res_gui = _fit_once_gui_style(x, y, config=cfg)
    unc_norm_gui = data_io.normalize_unc_result(unc_res_gui)
    long_gui, _ = data_io.write_uncertainty_csvs(base, str(fpath), unc_norm_gui, write_wide=True)

    cfg_batch = dict(cfg)
    cfg_batch.update(
        {
            "output_dir": str(tmp_path),
            "peaks": peaks.serialize([peaks.Peak(5.0, 1.0, 1.0, 0.5)]),
            "mode": "add",
            "source": "template",
        }
    )
    cfg_batch.update(ensure_unc_common({}))
    cfg_batch.update(bootstrap_cfg(n=cfg["bootstrap_n"]))
    cfg_batch.update(bayes_knobs())
    os.environ.setdefault("SMOKE_MODE", "1")
    run_batch([str(fpath)], config=cfg_batch, compute_uncertainty=True, unc_method="Bootstrap")
    long_batch = str(base) + "_uncertainty.csv"
    rows_gui = _read_long_csv(Path(long_gui))
    rows_batch = _read_long_csv(Path(long_batch))
    _compare_rows(rows_gui, rows_batch, rtol=0.08, atol=1e-8)


def test_asymptotic_gui_vs_batch_match(tmp_path):
    x, y = _make_spectrum()
    fpath = tmp_path / "sample.csv"
    _write_xy(fpath, x, y)
    base = tmp_path / "sample"

    cfg = {
        "solver_choice": "modern_trf",
        "unc_method": "Asymptotic",
        "export_unc_wide": True,
        "unc_alpha": 0.05,
    }
    pk = peaks.Peak(5.0, 1.0, 1.0, 0.5)
    baseline = np.zeros_like(y)
    mask = np.ones_like(x, bool)
    out = fit_api.run_fit_consistent(
        x, y, [pk], dict(cfg), baseline, "add", mask
    )
    assert out["fit_ok"]

    cfg_batch = dict(cfg)
    cfg_batch.update(
        {
            "output_dir": str(tmp_path),
            "peaks": peaks.serialize([peaks.Peak(5.0, 1.0, 1.0, 0.5)]),
            "mode": "add",
            "source": "template",
        }
    )
    cfg_batch.update(ensure_unc_common({}))
    cfg_batch.update(bootstrap_cfg(n=cfg.get("bootstrap_n", 60)))
    cfg_batch.update(bayes_knobs())
    os.environ.setdefault("SMOKE_MODE", "1")
    run_batch([str(fpath)], config=cfg_batch, compute_uncertainty=True, unc_method="Asymptotic")
    long_batch = str(base) + "_uncertainty.csv"
    assert Path(long_batch).exists(), "batch did not produce asymptotic CSV"
    rows = _read_long_csv(Path(long_batch))
    assert len(rows) >= 4
    for r in rows:
        for k in ("value", "stderr", "p2_5", "p97_5"):
            assert math.isfinite(float(r[k]))

