"""Batch processing orchestrator for Peakfit 3.x.

This module implements a lightweight batch runner that loads spectra matching
glob patterns, applies baseline correction, fits peaks using the selected
solver and writes a combined peak-table CSV. Optionally, per-spectrum trace
tables can also be emitted.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Sequence, List, Optional, Any

import csv
import os
import copy
import numpy as np
import math

if os.environ.get("SMOKE_MODE") == "1":  # pragma: no cover - environment safeguard
    os.environ.setdefault("MPLBACKEND", "Agg")

from core import data_io, models, peaks, signals, fit_api
from core.residuals import build_residual
from core.uncertainty import finite_diff_jacobian, bootstrap_ci, UncertaintyResult
from core.uncertainty_router import route_uncertainty


def _norm_jitter(v, default=0.02):
    """Normalize bootstrap jitter to a fractional value."""

    try:
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    except Exception:
        return float(default)


def _auto_seed(
    x: np.ndarray, y: np.ndarray, baseline: np.ndarray, max_peaks: int = 5
) -> List[peaks.Peak]:
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


def predict_full(
    x: np.ndarray,
    peaks_obj: Sequence[peaks.Peak],
    baseline: np.ndarray | None,
    mode: str,
    theta: np.ndarray,
) -> np.ndarray:
    """Evaluate the full model for ``theta`` on ``x``."""

    total = np.zeros_like(x, float)
    for i in range(len(peaks_obj)):
        c, h, fw, eta = theta[4 * i : 4 * (i + 1)]
        total += models.pseudo_voigt(x, h, c, fw, eta)
    if mode == "add" and baseline is not None:
        total = total + baseline
    return total


def run_batch(
    patterns: Iterable[str],
    config: dict,
    *,
    compute_uncertainty: bool = True,
    unc_method: str | None = None,
    progress=None,
    log=None,
    abort_event: Optional[Any] = None,
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
    if unc_workers < 0:
        unc_workers = 0

    out_dir = Path(
        config.get("output_dir", Path(config.get("peak_output", "peaks.csv")).parent)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.get("output_base")
    if base_name is None:
        peak_output_cfg = config.get("peak_output", out_dir / "batch_fit.csv")
        base_name = Path(peak_output_cfg).stem.replace("_fit", "")
    peak_output = out_dir / f"{base_name}_fit.csv"
    export_unc_wide = bool(config.get("export_unc_wide", False))
    # Choose uncertainty method: explicit arg > config aliases > default
    unc_choice = (
        unc_method
        or config.get("unc_method")
        or config.get("uncertainty_method")
        or config.get("uncertainty")
        or "asymptotic"
    )
    unc_method_canon = data_io.canonical_unc_label(unc_choice)
    if log:
        log(f"batch uncertainty method={unc_method_canon}")
        log(f"batch compute_uncertainty={bool(compute_uncertainty)}")

    records = []
    unc_rows = []
    ok = 0
    processed = 0

    for i, path in enumerate(files, 1):
        if abort_event is not None and getattr(abort_event, "is_set", lambda: False)():
            return {"aborted": True, "records": [], "reason": "user-abort"}
        if progress:
            progress(i, total, path)
        try:
            x, y = data_io.load_xy(path)
        except ValueError as exc:
            emsg = str(exc).lower()
            if (
                "more than 2" in emsg
                or "more than two" in emsg
                or "unsupported column" in emsg
            ):
                if log:
                    log(
                        f"{Path(path).name}: skipped (has >2 columns; mapping files not supported)"
                    )
                continue
            raise
        xmin = config.get("fit_xmin")
        xmax = config.get("fit_xmax")
        if xmin is not None and xmax is not None:
            lo, hi = sorted((float(xmin), float(xmax)))
            mask = (x >= lo) & (x <= hi)
        else:
            mask = np.ones_like(x, bool)

        method = str(base_cfg.get("method", "als")).lower()
        use_slice = bool(config.get("baseline_uses_fit_range", True))
        mask_eff = mask if use_slice else None
        if method == "als":
            if mask_eff is not None and np.any(mask_eff) and not np.all(mask_eff):
                x_sub = x[mask_eff]
                y_sub = y[mask_eff]
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
        elif method == "polynomial":
            deg = int(base_cfg.get("degree", 2))
            norm_x = bool(base_cfg.get("normalize_x", True))
            baseline = signals.polynomial_baseline(
                x, y, degree=deg, mask=mask_eff, normalize_x=norm_x
            )
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        if source == "auto":
            template = _auto_seed(x, y, baseline, max_peaks=auto_max)
        else:
            template = [
                peaks.Peak(
                    p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width
                )
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
            return_jacobian=True,
            return_predictors=True,
        )

        fitted = res["peaks_out"]
        theta = np.asarray(res["theta"], dtype=float)  # noqa: F841 - for debugging
        rmse = float(res["rmse"])
        if os.environ.get("IPF_SHADOW") == "1":
            x_fit = x[mask]
            y_fit = (y if mode == "add" else y - baseline)[mask]
            base_fit = baseline[mask] if mode == "add" else None
            resid_fn = build_residual(
                x_fit, y_fit, fitted, mode, base_fit, "linear", None
            )
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
                "baseline_uses_fit_range": bool(
                    config.get("baseline_uses_fit_range", True)
                ),
                **perf_extras,
                "bounds_center_lo": center_bounds[0],
                "bounds_center_hi": center_bounds[1],
                "bounds_fwhm_lo": fwhm_lo,
                "bounds_height_lo": 0.0,
                "bounds_height_hi": np.nan,
                "x_scale": np.nan,
            }
            baseline_method = str(base_cfg.get("method", "als")).lower()
            als_fields = {
                "als_lam": base_cfg.get("lam"),
                "als_p": base_cfg.get("p"),
                "als_niter": base_cfg.get("niter"),
                "als_thresh": base_cfg.get("thresh"),
            }
            if baseline_method != "als":
                als_fields = {k: np.nan for k in als_fields}
            # Polynomial metadata: values only for polynomial; NaN for ALS
            if baseline_method == "polynomial":
                poly_fields = {
                    "poly_degree": int(base_cfg.get("degree", 2)),
                    "poly_normalize_x": bool(base_cfg.get("normalize_x", True)),
                }
            else:
                poly_fields = {
                    "poly_degree": np.nan,
                    "poly_normalize_x": np.nan,
                }
            # Method label + fields
            rec["baseline_method"] = baseline_method
            rec.update(als_fields)
            rec.update(poly_fields)
            records.append(rec)
            local_records.append(rec)

        fit_csv = data_io.build_peak_table(local_records)
        with (out_dir / f"{Path(path).stem}_fit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            fh.write(fit_csv)

        unc_res = None
        if res["fit_ok"] and fitted and compute_uncertainty:
            try:
                theta_hat = np.asarray(res["theta"], float)
                x_fit = np.asarray(res.get("x_fit") or res.get("x"), float)
                y_fit = np.asarray(res.get("y_fit") or res.get("y"), float)
                peaks_obj = (
                    res.get("peaks_out")
                    or res.get("peaks")
                    or []
                )
                base_fit = res.get("baseline")
                mode = res.get("mode", "add")

                rj = res.get("residual_jac")
                residual_fn = res.get("residual_fn")
                if residual_fn is None and rj is not None:
                    residual_fn = lambda th: rj(th)[0]
                if residual_fn is None:
                    residual_fn = build_residual(
                        x_fit, y_fit, peaks_obj, mode, base_fit, "linear", None
                    )

                jac = res.get("jacobian")
                if jac is None and rj is not None:
                    jac = lambda th: rj(th)[1]
                if jac is None:
                    jac = finite_diff_jacobian(residual_fn, theta_hat)

                residual_vec = res.get("residual")
                if residual_vec is None:
                    residual_vec = residual_fn(theta_hat)
                residual_vec = np.asarray(residual_vec, float)

                model_eval = res.get("predict_full")
                def _predict_full_from_peaks(th, *, _x=x_fit, _peaks=peaks_obj, _base=base_fit, _mode=mode):
                    total = np.zeros_like(_x, float)
                    for idx in range(len(_peaks)):
                        c, h, fw, eta = th[4 * idx : 4 * (idx + 1)]
                        total += models.pseudo_voigt(_x, h, c, fw, eta)
                    if _mode == "add" and _base is not None:
                        total = total + _base
                    return total
                if model_eval is None:
                    model_eval = _predict_full_from_peaks

                fit_ctx = dict(res.get("fit_ctx") or {})
                solver_choice = str(
                    config.get("solver_choice", res.get("solver", "modern_trf"))
                )
                boot_solver = str(config.get("unc_boot_solver", solver_choice))
                has_lmfit = bool(res.get("has_lmfit", config.get("has_lmfit", False)))
                if boot_solver.lower().startswith("lmfit") and not has_lmfit:
                    boot_solver = solver_choice

                fit_ctx.update(
                    {
                        "residual_fn": residual_fn,
                        "predict_full": model_eval,
                        "x_all": x_fit,
                        "y_all": y_fit,
                        "baseline": base_fit,
                        "mode": mode,
                        "peaks": peaks_obj,
                        "peaks_out": peaks_obj,
                        "unc_workers": unc_workers,
                        "solver": boot_solver,
                        "bootstrap_jitter": _norm_jitter(
                            config.get("bootstrap_jitter", 0.02)
                        ),
                        "lmfit_share_fwhm": bool(config.get("lmfit_share_fwhm", False)),
                        "lmfit_share_eta": bool(config.get("lmfit_share_eta", False)),
                        "theta0": np.asarray(
                            res.get("theta0")
                            if res.get("theta0") is not None
                            else (res.get("p0") if res.get("p0") is not None else theta_hat),
                            float,
                        ),
                    }
                )

                from core import fit_api as _fit_api

                def _refit_wrapper(
                    theta_init,
                    locked_mask,
                    bounds,
                    x,
                    y,
                    res=res,
                    config=config,
                ):
                    peaks_in = res.get("peaks_out") or res.get("peaks") or []
                    cfg = copy.deepcopy(config)
                    mask = np.ones_like(x, bool)
                    try:
                        out = _fit_api.run_fit_consistent(
                            x,
                            y,
                            peaks_in,
                            cfg,
                            res.get("baseline"),
                            res.get("mode", "add"),
                            mask,
                            theta_init=theta_init,
                            locked_mask=locked_mask,
                            bounds=bounds,
                            baseline=res.get("baseline"),
                        )
                        return out["theta"]
                    except Exception:
                        try:
                            out = _fit_api.run_fit_consistent(
                                x,
                                y,
                                peaks_in,
                                cfg,
                                res.get("baseline"),
                                res.get("mode", "add"),
                                mask,
                            )
                            return out["theta"]
                        except Exception:
                            return np.asarray(theta_init, float)

                fit_ctx.update({"refit": _refit_wrapper})

                if str(unc_method_canon).lower() == "bootstrap":
                    bounds = res.get("bounds")
                    locked_mask = res.get("locked_mask")
                    alpha = float(config.get("unc_alpha", 0.05))
                    center_res = bool(config.get("unc_center_resid", True))
                    try:
                        n_boot = int(config.get("bootstrap_n", 200))
                    except Exception:
                        n_boot = 200
                    try:
                        seed_val = config.get("bootstrap_seed", 0)
                        seed_int = int(seed_val)
                    except Exception:
                        seed_int = 0
                    seed_val = None if seed_int == 0 else seed_int

                    jac_mat = jac(theta_hat) if callable(jac) else np.asarray(jac, float)
                    jac_mat = np.atleast_2d(np.asarray(jac_mat, float))

                    # Make Bootstrap deterministic when a seed is provided by running single-threaded.
                    workers_eff = None if (seed_val is not None) else unc_workers
                    fit_ctx["unc_workers"] = workers_eff
                    unc_res = bootstrap_ci(
                        theta=theta_hat,
                        residual=residual_vec,
                        jacobian=jac_mat,
                        predict_full=model_eval,
                        x_all=x_fit,
                        y_all=y_fit,
                        fit_ctx=fit_ctx,
                        bounds=bounds,
                        param_names=res.get("param_names"),
                        locked_mask=locked_mask,
                        n_boot=n_boot,
                        seed=seed_val,
                        workers=workers_eff,
                        alpha=alpha,
                        center_residuals=center_res,
                        return_band=True,
                    )
                    if isinstance(unc_res, UncertaintyResult):
                        diag = dict(unc_res.diagnostics)
                        unc_res = {
                            "method": unc_res.method,
                            "label": unc_res.label,
                            "method_label": unc_res.label,
                            "param_stats": unc_res.stats,
                            "diagnostics": diag,
                            "band": unc_res.band,
                            "alpha": diag.get("alpha", alpha),
                        }
                else:
                    unc_res = route_uncertainty(
                        unc_method_canon,
                        theta_hat=theta_hat,
                        residual_fn=residual_fn,
                        jacobian=jac,
                        model_eval=model_eval,
                        fit_ctx=fit_ctx,
                        x_all=x_fit,
                        y_all=y_fit,
                        workers=unc_workers,
                        seed=None,
                        n_boot=100,
                    )
            except Exception as exc:
                msg = str(exc)
                if "Unknown uncertainty method" in msg:
                    raise
                if log:
                    log(f"{Path(path).name}: uncertainty failed: {exc}")
                unc_res = None

        if unc_res is not None:
            stem = Path(path).stem

            unc_norm = data_io.normalize_unc_result(unc_res)
            method_lbl = data_io.canonical_unc_label(
                unc_norm.get("label") or unc_method_canon
            )
            unc_norm["label"] = method_lbl
            unc_norm["rmse"] = rmse if math.isfinite(rmse) else 0.0
            dof_raw = unc_norm.get("dof")
            try:
                dof_val = float(dof_raw)
            except Exception:
                dof_val = float("nan")
            if not math.isfinite(dof_val):
                dof_val = float(res.get("dof", 1)) if isinstance(res, dict) else 1.0
            unc_norm["dof"] = max(1, int(dof_val))

            data_io.write_uncertainty_txt(
                out_dir / f"{stem}_uncertainty.txt",
                unc_norm,
                peaks=fitted,
                method_label=method_lbl,
                file_path=path,
            )

            data_io.write_uncertainty_csvs(
                out_dir / stem,
                path,
                unc_norm,
                write_wide=export_unc_wide,
            )

            band = unc_norm.get("band")
            if band is not None:
                xb, lob, hib = band
                band_csv = out_dir / f"{stem}_uncertainty_band.csv"
                with band_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh, lineterminator="\n")
                    ci_pct = 95
                    try:
                        alpha = float(
                            unc_norm.get(
                                "alpha",
                                unc_norm.get("diagnostics", {}).get("alpha", 0.05),
                            )
                        )
                        ci_pct = int(round(100 * (1.0 - alpha)))
                    except Exception:
                        pass
                    writer.writerow(["x", f"y_lo{ci_pct}", f"y_hi{ci_pct}"])
                    for xi, lo, hi in zip(xb, lob, hib):
                        if abort_event is not None and getattr(abort_event, "is_set", lambda: False)():
                            return {"aborted": True, "records": [], "reason": "user-abort"}
                        if not (math.isfinite(float(xi)) and math.isfinite(float(lo)) and math.isfinite(float(hi))):
                            continue
                        writer.writerow([float(xi), float(lo), float(hi)])

            for row in data_io.iter_uncertainty_rows(path, unc_norm):
                if abort_event is not None and getattr(abort_event, "is_set", lambda: False)():
                    return {"aborted": True, "records": [], "reason": "user-abort"}
                unc_rows.append(row)
            if log:
                log(f"{Path(path).name}: uncertainty={method_lbl}")

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
        processed = i

    peak_csv = data_io.build_peak_table(records)
    with peak_output.open("w", encoding="utf-8", newline="") as fh:
        fh.write(peak_csv)
    if unc_rows:
        data_io.write_batch_uncertainty_long(out_dir, unc_rows)

    return ok, processed


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
    # Forward uncertainty selection to the batch runner
    um = (
        cfg.get("unc_method")
        or cfg.get("uncertainty_method")
        or cfg.get("uncertainty")
        or None
    )
    run_batch(
        patterns,
        cfg,
        compute_uncertainty=True,
        unc_method=um,
    )


def run(patterns: Iterable[str], config: dict, progress=None, log=None):
    """Compatibility wrapper using legacy signature."""
    um = (
        config.get("unc_method")
        or config.get("uncertainty_method")
        or config.get("uncertainty")
        or None
    )
    return run_batch(
        patterns,
        config,
        compute_uncertainty=True,
        unc_method=um,
        progress=progress,
        log=log,
    )
