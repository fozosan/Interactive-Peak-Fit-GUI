"""Batch processing orchestrator for Peakfit 3.x.

This module implements a lightweight batch runner that loads spectra matching
glob patterns, applies baseline correction, fits peaks using the selected
solver and writes a combined peak-table CSV. Optionally, per-spectrum trace
tables can also be emitted.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Sequence, List

import csv
import os
import copy
import numpy as np
import pandas as pd

if os.environ.get("SMOKE_MODE") == "1":  # pragma: no cover - environment safeguard
    os.environ.setdefault("MPLBACKEND", "Agg")

from core import data_io, models, peaks, signals, fit_api
from core.residuals import build_residual
from core import uncertainty as unc


def _auto_seed(x: np.ndarray, y: np.ndarray, baseline: np.ndarray, max_peaks: int = 5) -> List[peaks.Peak]:
    """Return up to ``max_peaks`` automatically seeded peaks."""

    sig = y - baseline
    if sig.size < 3:
        return []
    # simple local maxima detection
    mask = (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:])
    idx = np.where(mask)[0] + 1
    if idx.size == 0:
        return []
    prom = sig[idx]
    order = np.argsort(prom)[::-1][:max_peaks]
    idx = idx[order]
    span = float(x.max() - x.min())
    default_w = max(span * 0.05, float(np.mean(np.diff(np.sort(x)))) * 5.0)
    med = float(np.median(sig))
    found: List[peaks.Peak] = []
    for i in idx:
        h = max(float(sig[i] - med), 1e-6)
        found.append(peaks.Peak(float(x[i]), h, float(default_w), 0.5))
    found.sort(key=lambda p: p.center)
    return found


def run_batch(
    patterns: Iterable[str],
    config: dict,
    *,
    compute_uncertainty: bool = False,
    unc_method: str = "asymptotic",
    progress=None,
    log=None,
) -> tuple[int, int]:
    """Run the peak fitting pipeline over matching files.

    Parameters
    ----------
    patterns:
        Iterable of glob patterns. All matching files are processed.
    config:
        Dictionary describing the batch job. Supported keys include ``peaks``
        (list of peak dictionaries), ``solver`` (``classic`` | ``modern_vp`` |
        ``modern_trf`` | ``lmfit_vp``), ``mode`` (``add`` | ``subtract``), ``baseline``
        parameters, per-solver options, ``save_traces`` flag, ``peak_output``
        for the output CSV, ``source`` (``current`` | ``template`` | ``auto``)
        selecting the peak seeds, ``reheight`` to refresh heights per spectrum
        and ``auto_max`` controlling the maximum auto-seeded peaks.
    """

    if os.environ.get("SMOKE_MODE") == "1":
        config["perf_gpu"] = False
        config["perf_numba"] = False
        config["perf_max_workers"] = 0

    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise FileNotFoundError("no files matched patterns")
    total = len(files)

    base_template: Sequence[peaks.Peak] = [
        peaks.Peak(**p) for p in config.get("peaks", [])
    ]
    solver_name = config.get("solver", "classic")
    mode = config.get("mode", "add")
    base_cfg = config.get("baseline", {})
    save_traces = bool(config.get("save_traces", False))
    source = config.get("source", "template")
    reheight = bool(config.get("reheight", False))
    auto_max = int(config.get("auto_max", 5))
    unc_workers = int(config.get("unc_workers", 0))
    if unc_workers <= 0:
        unc_workers = int(config.get("perf_max_workers", 0)) or os.cpu_count() or 1

    out_dir = Path(config.get("output_dir", Path(config.get("peak_output", "peaks.csv")).parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.get("output_base")
    if base_name is None:
        peak_output_cfg = config.get("peak_output", out_dir / "batch_fit.csv")
        base_name = Path(peak_output_cfg).stem.replace("_fit", "")
    peak_output = out_dir / f"{base_name}_fit.csv"
    unc_output = out_dir / f"{base_name}_uncertainty.csv"

    out_dir = Path(config.get("output_dir", Path(config.get("peak_output", "peaks.csv")).parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.get("output_base")
    if base_name is None:
        peak_output_cfg = config.get("peak_output", out_dir / "batch_fit.csv")
        base_name = Path(peak_output_cfg).stem.replace("_fit", "")
    peak_output = out_dir / f"{base_name}_fit.csv"
    unc_output = out_dir / f"{base_name}_uncertainty.csv"

    records = []
    unc_rows = []
    ok = 0

    for i, path in enumerate(files, 1):
        if progress:
            progress(i, total, path)
        x, y = data_io.load_xy(path)
        xmin = config.get("fit_xmin")
        xmax = config.get("fit_xmax")
        if xmin is not None and xmax is not None:
            lo, hi = sorted((float(xmin), float(xmax)))
            mask = (x >= lo) & (x <= hi)
        else:
            mask = np.ones_like(x, bool)

        use_slice = bool(config.get("baseline_uses_fit_range", True))
        if use_slice and np.any(mask) and not np.all(mask):
            x_sub = x[mask]
            y_sub = y[mask]
            z_sub = signals.als_baseline(
                y_sub,
                lam=base_cfg.get("lam", 1e5),
                p=base_cfg.get("p", 0.001),
                niter=base_cfg.get("niter", 10),
                tol=base_cfg.get("thresh", 0.0),
            )
            baseline = np.interp(x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
        else:
            baseline = signals.als_baseline(
                y,
                lam=base_cfg.get("lam", 1e5),
                p=base_cfg.get("p", 0.001),
                niter=base_cfg.get("niter", 10),
                tol=base_cfg.get("thresh", 0.0),
            )

        if source == "auto":
            template = _auto_seed(x, y, baseline, max_peaks=auto_max)
        else:
            template = [
                peaks.Peak(p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width)
                for p in base_template
            ]

        peaks_in = copy.deepcopy(template)
        seed = (
            abs(hash(str(Path(path).resolve()))) & 0xFFFFFFFF
            if config.get("perf_seed_all", False)
            else None
        )
        res = fit_api.run_fit_consistent(
            x,
            y,
            peaks_in,
            config,
            baseline,
            mode,
            mask,
            reheight=reheight,
            rng_seed=seed,
            verbose=bool(log),
            return_jacobian=compute_uncertainty,
            return_predictors=compute_uncertainty,
        )

        fitted = res["peaks_out"]
        theta = np.asarray(res["theta"], dtype=float)  # noqa: F841 - for debugging
        rmse = float(res["rmse"])
        if os.environ.get("IPF_SHADOW") == "1":
            x_fit = x[mask]
            y_fit = (y if mode == "add" else y - baseline)[mask]
            base_fit = baseline[mask] if mode == "add" else None
            resid_fn = build_residual(x_fit, y_fit, fitted, mode, base_fit, "linear", None)
            r = resid_fn(theta)
            rmse_shadow = float(np.sqrt(np.mean(r * r))) if r.size else float("nan")
            if abs(rmse_shadow - rmse) > 1e-8 and log:
                log(f"shadow rmse diff {rmse_shadow - rmse}")
        areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in fitted]
        total = sum(areas) or 1.0
        perf_extras = {
            "perf_numba": bool(config.get("perf_numba", False)),
            "perf_gpu": bool(config.get("perf_gpu", False)),
            "perf_cache_baseline": bool(config.get("perf_cache_baseline", True)),
            "perf_seed_all": bool(config.get("perf_seed_all", False)),
            "perf_max_workers": int(config.get("perf_max_workers", 0)),
        }
        solver_opts = dict(config.get(solver_name, {}))
        center_bounds = (
            (float(x[0]), float(x[-1]))
            if (
                solver_opts.get("centers_in_window")
                or solver_opts.get("bound_centers_to_window")
            )
            else (np.nan, np.nan)
        )
        med_dx = float(np.median(np.diff(np.sort(x)))) if x.size > 1 else 0.0
        fwhm_lo = solver_opts.get("min_fwhm", max(1e-6, 2.0 * med_dx))
        fit_lo = float(config.get("fit_xmin", x[0]))
        fit_hi = float(config.get("fit_xmax", x[-1]))
        local_records = []
        for idx, (p, area) in enumerate(zip(fitted, areas), start=1):
            rec = {
                "file": Path(path).name,
                "peak": idx,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "lock_width": p.lock_width,
                "lock_center": p.lock_center,
                "area": area,
                "area_pct": 100.0 * area / total,
                "rmse": rmse,
                "fit_ok": bool(res["fit_ok"]),
                "mode": mode,
                "als_lam": base_cfg.get("lam"),
                "als_p": base_cfg.get("p"),
                "fit_xmin": fit_lo,
                "fit_xmax": fit_hi,
                "solver_choice": solver_name,
                "solver_loss": config.get("solver_loss", np.nan),
                "solver_weight": config.get("solver_weight", np.nan),
                "solver_fscale": config.get("solver_fscale", np.nan),
                "solver_maxfev": config.get("solver_maxfev", np.nan),
                "solver_restarts": config.get("solver_restarts", np.nan),
                "solver_jitter_pct": config.get("solver_jitter_pct", np.nan),
                "use_baseline": True,
                "baseline_mode": mode,
                "baseline_uses_fit_range": bool(config.get("baseline_uses_fit_range", True)),
                "als_niter": base_cfg.get("niter"),
                "als_thresh": base_cfg.get("thresh"),
                **perf_extras,
                "bounds_center_lo": center_bounds[0],
                "bounds_center_hi": center_bounds[1],
                "bounds_fwhm_lo": fwhm_lo,
                "bounds_height_lo": 0.0,
                "bounds_height_hi": np.nan,
                "x_scale": np.nan,
            }
            records.append(rec)
            local_records.append(rec)

        fit_csv = data_io.build_peak_table(local_records)
        with (out_dir / f"{Path(path).stem}_fit.csv").open("w", encoding="utf-8", newline="") as fh:
            fh.write(fit_csv)

        unc_res = None
        if compute_uncertainty and res["fit_ok"]:
            try:
                if "boot" in unc_method.lower():
                    unc_res = unc.bootstrap_ci(fit_ctx=res, n_boot=100, workers=unc_workers)
                elif "bayes" in unc_method.lower() or "mcmc" in unc_method.lower():
                    unc_res = unc.bayesian_ci(fit_ctx=res)
                else:
                    unc_res = unc.asymptotic_ci(
                        res["theta"], res["residual_fn"], res["jacobian"], res["ymodel_fn"]
                    )
            except Exception:
                unc_res = None

        if compute_uncertainty and unc_res is not None:
            stem = Path(path).stem
            data_io.write_uncertainty_txt(
                out_dir / f"{stem}_uncertainty.txt",
                unc_res,
                peaks=fitted,
                method_label=unc_method,
            )
            data_io.write_uncertainty_csv(
                out_dir / f"{stem}_uncertainty.csv",
                unc_res,
                peaks=fitted,
                method_label=unc_method,
            )
            band = getattr(unc_res, "band", None)
            if band is not None:
                xb, lob, hib = band
                with (out_dir / f"{stem}_uncertainty_band.csv").open("w", newline="", encoding="utf-8") as fh:
                    bw = csv.writer(fh, lineterminator="\n")
                    bw.writerow(["x", "y_lo95", "y_hi95"])
                    for xi, lo, hi in zip(xb, lob, hib):
                        bw.writerow([xi, lo, hi])
            for row in data_io._iter_param_rows(unc_res, fitted, unc_method):
                unc_rows.append(
                    {
                        "file": Path(path).name,
                        "peak": row.get("peak"),
                        "param": row.get("param"),
                        "value": row.get("est"),
                        "stderr": row.get("sd"),
                        "ci_lo": row.get("p2_5"),
                        "ci_hi": row.get("p97_5"),
                        "method": unc_method,
                        "rmse": rmse,
                        "dof": np.nan,
                    }
                )

        trace_path = None
        if save_traces:
            trace_csv = data_io.build_trace_table(x, y, baseline, fitted)
            trace_path = out_dir / f"{Path(path).stem}_trace.csv"
            with trace_path.open("w", encoding="utf-8", newline="") as fh:
                fh.write(trace_csv)
        if log:
            msg = f"{Path(path).name}: {'ok' if res['fit_ok'] else 'fail'} rmse={rmse:.3g}"
            if trace_path:
                msg += f" {trace_path}"
            log(msg)
        if res["fit_ok"]:
            ok += 1

    peak_csv = data_io.build_peak_table(records)
    with peak_output.open("w", encoding="utf-8", newline="") as fh:
        fh.write(peak_csv)
    if compute_uncertainty and unc_rows:
        data_io.write_dataframe(pd.DataFrame(unc_rows), unc_output)

    return ok, total


def run_from_dir(
    *,
    input_dir: str,
    pattern: str = "*.csv",
    output_dir: str,
    source_mode: str = "current",
    reheight: bool = False,
    seed: int | None = None,
    workers: int = 0,
    perf_overrides: dict | None = None,
    files_filter: list[str] | None = None,
):
    """Convenience wrapper used by smoke tests.

    Parameters mirror :func:`run` but expose a directory+pattern interface. Only
    a very small subset of configuration options is supported. This helper
    exists purely for test and tooling purposes and does not aim to cover the
    full batch runner feature set.
    """

    if files_filter:
        patterns = [os.path.join(input_dir, f) for f in files_filter]
    else:
        patterns = [os.path.join(input_dir, pattern)]

    cfg = {
        "peaks": [],
        "solver": "modern_vp",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "output_dir": output_dir,
        "output_base": "batch",
        "save_traces": True,
        "source": source_mode,
        "reheight": reheight,
        "perf_seed_all": True,
        "perf_max_workers": workers,
    }

    if perf_overrides:
        cfg.update(perf_overrides)

    if os.environ.get("SMOKE_MODE") == "1":
        cfg["perf_gpu"] = False
        cfg["perf_numba"] = False
        cfg["perf_max_workers"] = 0

    run_batch(patterns, cfg)


def run(patterns: Iterable[str], config: dict, progress=None, log=None):
    """Compatibility wrapper using legacy signature."""
    return run_batch(
        patterns,
        config,
        compute_uncertainty=True,
        unc_method=config.get("unc_method", "asymptotic"),
        progress=progress,
        log=log,
    )

