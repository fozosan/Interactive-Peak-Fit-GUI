"""Batch processing orchestrator for Peakfit 3.x.

This module implements a lightweight batch runner that loads spectra matching
glob patterns, applies baseline correction, fits peaks using the selected
solver and writes a combined peak-table CSV. Optionally, per-spectrum trace
tables can also be emitted.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Sequence, List, Any
from typing import Optional

import csv
import os
import copy
import numpy as np
import math
import logging
from contextlib import nullcontext

if os.environ.get("SMOKE_MODE") == "1":  # pragma: no cover - environment safeguard
    os.environ.setdefault("MPLBACKEND", "Agg")

import core.data_io as data_io
from core import models, peaks, signals, fit_api
from core.residuals import build_residual, finite_diff_jacobian
from core.uncertainty import bootstrap_ci, UncertaintyResult
from core.uncertainty_router import route_uncertainty
from infra import performance


logger = logging.getLogger(__name__)


def _tp_limits_ctx_for_config(config: dict):
    """
    Robust BLAS/threadpool limiter for batch path.
    Returns a real context object (threadpool_limits or nullcontext), never a generator.
    """

    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except Exception:
        threadpool_limits = None  # type: ignore

    strategy = str(config.get("perf_parallel_strategy", "outer"))
    try:
        _bt = int(config.get("perf_blas_threads", 0) or 0)
    except Exception:
        _bt = 0
    limit = 1 if strategy == "outer" else (_bt if _bt > 0 else None)
    if threadpool_limits is not None and limit is not None:
        try:
            return threadpool_limits(limits=limit)
        except Exception:
            return nullcontext()
    return nullcontext()


def _norm_jitter(v, default=0.02):
    """Normalize bootstrap jitter to a fractional value."""

    try:
        f = float(v)
    except Exception:
        return float(default)
    if f < 0:
        f = 0.0
    if f > 1.5:
        f = f / 100.0
    if f > 1.0:
        f = 1.0
    return f


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
        if len(baseline) != len(x):
            raise ValueError(f"baseline length mismatch: x={len(x)} baseline={len(baseline)}")
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
    abort_on_uncertainty_failure: bool = False,
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
    # If a generic "solver_options" block is provided, merge into the
    # per-solver config so run_fit_consistent → pack_theta_bounds sees caps.
    if "solver_options" in config:
        cfg_opts = dict(config.get(solver_name, {}))
        cfg_opts.update(dict(config.get("solver_options") or {}))
        config[solver_name] = cfg_opts
    mode = config.get("mode", "add")
    base_cfg = config.get("baseline", {})
    save_traces = bool(config.get("save_traces", False))
    source = config.get("source", "template")
    reheight = bool(config.get("reheight", False))
    auto_max = int(config.get("auto_max", 5))
    cfg_perf = performance.get_parallel_config()
    performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)

    maxcpu = max(1, (os.cpu_count() or 1))

    def _parse_workers(value) -> int:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return 0

    unc_req_cfg = _parse_workers(
        config.get(
            "perf_unc_workers",
            config.get("unc_workers", config.get("perf_max_workers", 0)),
        )
    )
    if unc_req_cfg <= 0:
        unc_workers = min(maxcpu, max(1, int(cfg_perf.unc_workers)))
    else:
        unc_workers = max(1, min(unc_req_cfg, maxcpu))

    band_req_cfg = _parse_workers(
        config.get(
            "unc_band_workers",
            unc_req_cfg if unc_req_cfg > 0 else cfg_perf.unc_workers,
        )
    )
    if band_req_cfg <= 0:
        unc_band_workers = unc_workers
    else:
        unc_band_workers = max(1, min(band_req_cfg, maxcpu))

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
    export_cfg = dict(config.get("export", {}) or {})
    unc_section = dict(config.get("uncertainty", {}) or {})

    def _method_key(val) -> str:
        if isinstance(val, dict):
            val = val.get("method", "")
        if val is None:
            return ""
        return str(val).strip().lower()

    unc_method_arg = _method_key(unc_method)
    want_unc_config = bool(export_cfg.get("include_uncertainty", False))
    method_requested = bool(unc_method_arg)
    if compute_uncertainty is None:
        compute_uncertainty_flag = bool(want_unc_config or method_requested)
    else:
        compute_uncertainty_flag = bool(compute_uncertainty)
    need_band = bool(export_cfg.get("include_band", False)) or bool(
        unc_section.get("return_band", False)
    )
    band_pref_bootstrap = bool(config.get("ui_band_pref_bootstrap", True)) or need_band
    compute_uncertainty = compute_uncertainty_flag
    expected_unc = want_unc_config or (compute_uncertainty_flag and method_requested)
    # Choose uncertainty method: explicit arg > config aliases > default
    unc_choice = (
        unc_method_arg
        or _method_key(config.get("unc_method"))
        or _method_key(config.get("uncertainty_method"))
        or _method_key(unc_section)
        or "asymptotic"
    )
    unc_method_canon = data_io.canonical_unc_label(unc_choice)
    bootstrap_n_cfg: Optional[int] = None
    if str(unc_method_canon).lower() == "bootstrap":
        if "bootstrap_n" not in config:
            raise KeyError("bootstrap_n missing in batch config")
        try:
            bootstrap_n_cfg = int(config["bootstrap_n"])
        except Exception as exc:
            raise ValueError(
                f"invalid bootstrap_n: {config.get('bootstrap_n')!r}"
            ) from exc
        if bootstrap_n_cfg <= 0:
            raise ValueError(
                f"bootstrap_n must be > 0 (got {bootstrap_n_cfg})"
            )
    perf_seed_all = bool(config.get("perf_seed_all", False))
    perf_seed = None
    if perf_seed_all:
        try:
            perf_seed = int(str(config.get("perf_seed", "")).strip())
        except Exception:
            perf_seed = None
    if log:
        log(f"batch uncertainty method={unc_method_canon}")
        log(f"batch compute_uncertainty={bool(compute_uncertainty)}")

    records = []
    unc_rows = []
    uncertainty_failures = []
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
        unc_error_msg = None
        if res["fit_ok"] and fitted and compute_uncertainty:
            try:
                theta_hat = np.asarray(res["theta"], float)
                x_all = res.get("x")
                if x_all is None:
                    x_all = x
                x_all = np.asarray(x_all, float)

                y_all = res.get("y")
                if y_all is None:
                    y_all = y
                y_all = np.asarray(y_all, float)

                x_fit = res.get("x_fit")
                if x_fit is None:
                    x_fit = x_all
                x_fit = np.asarray(x_fit, float)

                y_fit = res.get("y_fit")
                if y_fit is None:
                    y_fit = y_all
                y_fit = np.asarray(y_fit, float)
                peaks_obj = (
                    res.get("peaks_out")
                    or res.get("peaks")
                    or []
                )
                base_fit = res.get("baseline")
                mode = res.get("mode", "add")
                add_mode = mode == "add"

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

                # Predictor: prefer solver-provided fit-window function; otherwise build locally on x_fit.
                predict_fit_fn = res.get("predict_fit")

                def _predict_fit_local(
                    th,
                    *,
                    _x=x_fit,
                    _peaks=peaks_obj,
                    _base=base_fit,
                    _add=add_mode,
                ):
                    total = np.zeros_like(_x, float)
                    t = np.asarray(th, float).ravel()
                    for i in range(len(_peaks)):
                        j = 4 * i
                        c, h, fw, eta = t[j : j + 4]
                        total += models.pseudo_voigt(_x, h, c, fw, eta)
                    if _add and _base is not None:
                        total = total + _base
                    return total

                if not callable(predict_fit_fn):
                    predict_fit_fn = _predict_fit_local
                else:
                    try:
                        _probe = np.asarray(
                            predict_fit_fn(np.asarray(theta_hat, float)), float
                        ).reshape(-1)
                        if int(_probe.size) != int(x_fit.size):
                            predict_fit_fn = _predict_fit_local
                    except Exception:
                        predict_fit_fn = _predict_fit_local

                predict_full_fn = res.get("predict_full")
                if callable(predict_full_fn):
                    try:
                        _probe_full = np.asarray(
                            predict_full_fn(np.asarray(theta_hat, float)), float
                        ).reshape(-1)
                        if int(_probe_full.size) != int(x_all.size):
                            raise ValueError("predict_full size mismatch")
                    except Exception:
                        predict_full_fn = None

                # Build predictor for Bootstrap; if not provided, bind it to the *fit* grid to match residuals/band.
                if not callable(predict_full_fn):
                    base_fit_arr = np.asarray(base_fit, float) if (
                        add_mode and base_fit is not None
                    ) else None

                    def _predict_full_bootstrap(
                        th,
                        *,
                        _x=x_fit,
                        _peaks=peaks_obj,
                        _base=base_fit_arr,
                        _add=add_mode,
                    ):
                        total = np.zeros_like(_x, float)
                        t = np.asarray(th, float).ravel()
                        for i in range(len(_peaks)):
                            j = 4 * i
                            c, h, fw, eta = t[j : j + 4]
                            total += models.pseudo_voigt(_x, h, c, fw, eta)
                        if _add and _base is not None:
                            total = total + _base
                        return total

                    predict_full_fn = _predict_full_bootstrap

                fit_ctx = dict(res.get("fit_ctx") or {})
                solver_choice = (
                    config.get("solver_choice")
                    or config.get("solver")
                    or res.get("solver")
                    or "modern_trf"
                )
                solver_choice = str(solver_choice)
                # Normalize & seed bootstrap defaults for parity with GUI
                fit_ctx.setdefault("bootstrap_residual_mode", "raw")
                fit_ctx.setdefault("relabel_by_center", True)
                fit_ctx.setdefault("center_residuals", True)
                # Hand the solved peak list and constraints from THIS fit
                fit_ctx["peaks_out"] = peaks_obj
                # Forward exact constraints and start used by the solver so bootstrap refits respect them
                bounds_exact = res.get("bounds")
                if bounds_exact is not None:
                    fit_ctx["bounds"] = bounds_exact
                locked_exact = res.get("locked_mask")
                if locked_exact is not None:
                    fit_ctx["locked_mask"] = locked_exact
                fit_ctx["theta0"] = np.asarray(theta_hat, float)
                if not str(solver_choice).lower().startswith("lmfit"):
                    fit_ctx.pop("lmfit_share_fwhm", None)
                    fit_ctx.pop("lmfit_share_eta", None)
                boot_solver = (
                    config.get("unc_boot_solver")
                    or solver_choice
                    or config.get("solver")
                    or "modern_vp"
                )
                boot_solver = str(boot_solver)
                has_lmfit = bool(res.get("has_lmfit", config.get("has_lmfit", False)))
                if boot_solver.lower().startswith("lmfit") and not has_lmfit:
                    boot_solver = solver_choice

                jitter_frac = _norm_jitter(config.get("bootstrap_jitter", 0.02))
                centers_ref = [float(p.center) for p in peaks_obj] if peaks_obj else []
                # Make the chosen bootstrap solver visible to inner refits.
                fit_ctx["unc_boot_solver"] = str(boot_solver)

                # Also mirror constraints at the top level when invoking bootstrap_ci.
                top_bounds = bounds_exact
                top_locked = locked_exact

                fit_ctx.update(
                    {
                        "residual_fn": residual_fn,
                        "predict_full": predict_full_fn,
                        "predict_fit": predict_fit_fn,
                        "x_all": x_fit,
                        "y_all": y_fit,
                        "x_fit": x_fit,
                        "y_fit": y_fit,
                        "baseline": base_fit,
                        "mode": mode,
                        "peaks": peaks_obj,
                        # set solver once to the chosen bootstrap engine
                        "solver": boot_solver,
                        # Make per-peak caps available to bootstrap's bounds derivation
                        "solver_options": dict(config.get("solver_options", {})),
                        "bootstrap_jitter": jitter_frac,
                        "lmfit_share_fwhm": bool(config.get("lmfit_share_fwhm", False)),
                        "lmfit_share_eta": bool(config.get("lmfit_share_eta", False)),
                        "centers_ref": centers_ref,
                        "relabel_by_center": True,
                        # Parity with GUI: never use linearized fallback path
                        "allow_linear_fallback": False,
                        "strict_refit": True,
                    }
                )
                try:
                    _blas_cfg = int(config.get("perf_blas_threads", 0) or 0)
                except Exception:
                    _blas_cfg = 0
                fit_ctx.update(
                    {
                        "perf_parallel_strategy": str(
                            config.get("perf_parallel_strategy", "outer")
                        ),
                        "perf_blas_threads": _blas_cfg,
                    }
                )
                workers_eff = int(unc_workers)
                band_workers_eff = int(unc_band_workers)
                fit_ctx.update({
                    "unc_workers": workers_eff if workers_eff > 0 else None,
                    "unc_band_workers": (
                        band_workers_eff if band_workers_eff > 0 else None
                    ),
                    "unc_use_gpu": bool(config.get("unc_use_gpu", False)),
                })

                # keep 3.9-compatible typing
                n_boot: Optional[int] = None

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
                    peaks_in = copy.deepcopy(
                        res.get("peaks_out") or res.get("peaks") or []
                    )
                    cfg = copy.deepcopy(config)
                    fit_mask = np.ones_like(x, bool)

                    # Inject jittered theta into a copy of peaks so run_fit_consistent uses it as p0
                    peaks_start = copy.deepcopy(peaks_in)
                    try:
                        if peaks_start and 4 * len(peaks_start) == int(np.asarray(theta_init).size):
                            t = np.asarray(theta_init, float).ravel()
                            for i, pk in enumerate(peaks_start):
                                j = 4 * i
                                pk.center = float(t[j + 0])
                                pk.height = float(t[j + 1])
                                pk.fwhm = float(t[j + 2])
                                pk.eta = float(t[j + 3])
                    except Exception:
                        # Fall back to original peaks_in if anything goes sideways
                        peaks_start = peaks_in

                    # Ensure the refit uses the requested bootstrap solver (both keys for safety)
                    solver_choice_local = str(fit_ctx.get("unc_boot_solver", "")) or str(
                        config.get("unc_boot_solver")
                        or config.get("solver_choice")
                        or "modern_vp"
                    )
                    cfg["solver_choice"] = solver_choice_local
                    cfg["solver"] = solver_choice_local
                    # Mirror generic solver_options into the active solver's cfg for refits
                    if "solver_options" in config:
                        _extra = dict(config.get(solver_choice_local, {}))
                        _extra.update(dict(config.get("solver_options") or {}))
                        cfg[solver_choice_local] = _extra
                    if solver_choice_local.lower() in ("lmfit_vp", "lmfit-vp", "lmfit"):
                        cfg["strict_refit"] = True
                        cfg["relabel_by_center"] = True
                    else:
                        cfg.pop("strict_refit", None)
                        cfg["relabel_by_center"] = True

                    # Single call; consider it success only if theta is finite and sized correctly
                    try:
                        out = _fit_api.run_fit_consistent(
                            x,
                            y,
                            peaks_start,
                            cfg,
                            baseline=res.get("baseline"),
                            mode=res.get("mode", "add"),
                            fit_mask=fit_mask,
                            theta_init=np.asarray(theta_init, float),
                            locked_mask=locked_mask,
                            bounds=bounds,
                        )
                    except TypeError:
                        try:
                            out = _fit_api.run_fit_consistent(
                                x,
                                y,
                                peaks_start,
                                cfg,
                                baseline=res.get("baseline"),
                                mode=res.get("mode", "add"),
                                fit_mask=fit_mask,
                            )
                        except TypeError:
                            try:
                                out = _fit_api.run_fit_consistent(x, y, peaks_start, cfg)
                            except Exception:
                                th_fallback = np.asarray(theta_init, float)
                                return th_fallback, np.all(np.isfinite(th_fallback))
                        except Exception:
                            th_fallback = np.asarray(theta_init, float)
                            return th_fallback, np.all(np.isfinite(th_fallback))
                    except Exception:
                        th_fallback = np.asarray(theta_init, float)
                        return th_fallback, np.all(np.isfinite(th_fallback))

                    # Normalize and validate output
                    try:
                        if isinstance(out, dict):
                            theta_candidate = out.get("theta")
                            if theta_candidate is None:
                                for key in ("theta_out", "params", "theta_best"):
                                    theta_candidate = out.get(key)
                                    if theta_candidate is not None:
                                        break
                            if theta_candidate is None:
                                theta_candidate = theta_init
                            theta_out = np.asarray(theta_candidate, float).ravel()
                            ok_flag = out.get("fit_ok")
                            if ok_flag is None:
                                ok_flag = out.get("ok")
                            if ok_flag is None:
                                ok_flag = out.get("success")
                            if ok_flag is None:
                                ok_flag = out.get("converged", True)
                        else:
                            theta_out = np.asarray(out, float).ravel()
                            ok_flag = True
                        theta_init_flat = np.asarray(theta_init, float).ravel()
                        theta_out_flat = np.asarray(theta_out, float).ravel()
                        if theta_out_flat.size >= theta_init_flat.size:
                            theta_use = theta_out_flat[: theta_init_flat.size]
                            size_ok = True
                        else:
                            theta_use = theta_init_flat
                            size_ok = False
                            try:
                                logger.debug(
                                    "bootstrap refit size mismatch: out=%d init=%d", theta_out_flat.size, theta_init_flat.size
                                )
                            except Exception:
                                pass
                        if size_ok:
                            theta_use = np.where(
                                np.isfinite(theta_use),
                                theta_use,
                                theta_init_flat,
                            )
                        ok = size_ok and np.all(np.isfinite(theta_use))
                        if not ok:
                            try:
                                logger.debug(
                                    "bootstrap refit bad theta: ok_flag=%r size_out=%d size_init=%d theta=%s",
                                    ok_flag,
                                    theta_out_flat.size,
                                    theta_init_flat.size,
                                    theta_out_flat,
                                )
                            except Exception:
                                pass
                        return theta_use, ok
                    except Exception:
                        th_fallback = np.asarray(theta_init, float)
                        return th_fallback, np.all(np.isfinite(th_fallback))

                fit_ctx.update({"refit": _refit_wrapper})

                workers_eff = int(unc_workers)
                band_workers_eff = int(unc_band_workers)
                fit_ctx["unc_workers"] = workers_eff if workers_eff > 0 else None
                fit_ctx["unc_band_workers"] = (
                    band_workers_eff if band_workers_eff > 0 else None
                )

                if str(unc_method_canon).lower() == "bootstrap":
                    if bootstrap_n_cfg is None:
                        raise KeyError("bootstrap_n missing in batch config")
                    n_boot = int(bootstrap_n_cfg)
                    bounds = res.get("bounds")
                    locked_mask = res.get("locked_mask")
                    alpha = float(config.get("unc_alpha", 0.05))
                    center_res = bool(config.get("unc_center_resid", True))
                    seed_val = perf_seed if perf_seed_all and (perf_seed is not None) else None

                    jac_mat = jac(theta_hat) if callable(jac) else np.asarray(jac, float)
                    jac_mat = np.atleast_2d(np.asarray(jac_mat, float))
                    jac = jac_mat

                    fit_ctx["abort_event"] = abort_event
                    strategy = str(config.get("perf_parallel_strategy", "outer"))
                    try:
                        _bt_raw = int(config.get("perf_blas_threads", 0) or 0)
                    except Exception:
                        _bt_raw = 0
                    blas_effective = 1 if strategy == "outer" else (_bt_raw if _bt_raw > 0 else "lib")
                    workers_arg = workers_eff if workers_eff > 0 else None

                    perf_line = (
                        f"[DEBUG] perf: fit_workers={cfg_perf.fit_workers} unc_workers={workers_eff} "
                        f"blas_threads={blas_effective} seed_all={perf_seed_all} seed={seed_val}"
                    )
                    logger.debug(perf_line)
                    if log:
                        log(perf_line)

                    # Run bootstrap with the configured solver; NO fallbacks. If it fails, surface it.
                    fit_ctx_local = dict(fit_ctx)
                    fit_ctx_local["unc_boot_solver"] = str(boot_solver)
                    fit_ctx_local["solver"] = str(boot_solver)
                    # Always force strict non-linear refits for bootstrap (matches GUI stability)
                    fit_ctx_local["strict_refit"] = True
                    if "boot_studentize" in config:
                        fit_ctx_local["boot_studentize"] = config.get("boot_studentize")
                    if "boot_resampling" in config:
                        fit_ctx_local["boot_resampling"] = config.get("boot_resampling")
                    if "boot_wild_weights" in config:
                        fit_ctx_local["boot_wild_weights"] = config.get("boot_wild_weights")
                    if "boot_center_clamp_mode" in config:
                        fit_ctx_local["boot_center_clamp_mode"] = config.get("boot_center_clamp_mode")
                    if "boot_center_clamp_value" in config:
                        fit_ctx_local["boot_center_clamp_value"] = config.get("boot_center_clamp_value")
                    with _tp_limits_ctx_for_config(config):
                        performance.apply_global_seed(seed_val, perf_seed_all)
                        try:
                            # IMPORTANT: align everything to the fit grid so Y, residuals and band share the same length
                            unc_res = bootstrap_ci(
                                theta=theta_hat,
                                residual=residual_vec,
                                jacobian=jac,
                                predict_full=predict_full_fn,
                                x_all=x_fit,
                                y_all=y_fit,
                                x_eval=x_fit,
                                fit_ctx=fit_ctx_local,
                                bounds=top_bounds if top_bounds is not None else bounds_exact,
                                param_names=res.get("param_names"),
                                locked_mask=top_locked if top_locked is not None else locked_exact,
                                n_boot=n_boot,
                                seed=seed_val,
                                workers=workers_arg,
                                alpha=alpha,
                                center_residuals=center_res,
                                return_band=band_pref_bootstrap,
                                jitter=jitter_frac,
                            )
                        except Exception as _e:
                            ctx = (
                                f" (solver={boot_solver}, jitter={jitter_frac:.3f}, "
                                f"workers={workers_arg}, band_workers={unc_band_workers})"
                            )
                            raise type(_e)(f"{_e}{ctx}").with_traceback(_e.__traceback__)
                    if isinstance(unc_res, dict):
                        diag = (unc_res.get("diagnostics") or {})
                    else:
                        diag = getattr(unc_res, "diagnostics", {}) or {}
                    errs = (diag.get("refit_errors") or [])[:5]
                    if errs:
                        msg = "; ".join(map(str, errs))
                        logger.info("bootstrap refit errors (first few): %s", msg)
                        if log:
                            log(f"bootstrap refit errors (first few): {msg}")
                    if isinstance(unc_res, UncertaintyResult):
                        diag = dict(unc_res.diagnostics)
                        unc_res = {
                            "method": unc_res.method,
                            "label": unc_res.label,
                            "method_label": unc_res.label,
                            "stats": unc_res.stats,
                            "diagnostics": diag,
                            "band": unc_res.band,
                            "alpha": diag.get("alpha", alpha),
                        }
                else:
                    if "bayes" in str(unc_method_canon).lower():
                        try:
                            walkers_cfg_raw = config["bayes_walkers"]
                        except KeyError as exc:
                            raise KeyError(
                                "Missing required Bayesian knob: bayes_walkers"
                            ) from exc
                        if walkers_cfg_raw is None:
                            raise ValueError("bayes_walkers must be ≥ 0 (got None)")
                        try:
                            walkers_cfg = int(walkers_cfg_raw)
                        except Exception as exc:
                            raise ValueError(
                                f"Invalid bayes_walkers: {walkers_cfg_raw!r}"
                            ) from exc
                        if walkers_cfg < 0:
                            raise ValueError(
                                f"bayes_walkers must be ≥ 0 (got {walkers_cfg})"
                            )
                        try:
                            burn_cfg = int(config["bayes_burn"])
                            steps_cfg = int(config["bayes_steps"])
                            thin_cfg = int(config["bayes_thin"])
                        except KeyError as exc:
                            raise KeyError(
                                f"Missing required Bayesian knob: {exc.args[0]}"
                            ) from exc
                        except Exception as exc:
                            raise ValueError(
                                "Invalid Bayesian knobs in config: burn/steps/thin"
                            ) from exc
                        if burn_cfg < 0:
                            raise ValueError(f"bayes_burn must be ≥ 0 (got {burn_cfg})")
                        if steps_cfg <= 0:
                            raise ValueError(f"bayes_steps must be > 0 (got {steps_cfg})")
                        if thin_cfg <= 0:
                            raise ValueError(f"bayes_thin must be > 0 (got {thin_cfg})")
                        fit_ctx.update({
                            "bayes_diagnostics": bool(
                                config.get(
                                    "bayes_diagnostics", config.get("bayes_diag", False)
                                )
                            ),
                            "unc_band_workers": band_workers_eff
                            if band_workers_eff > 0
                            else None,
                            "bayes_band_enabled": bool(
                                config.get("bayes_band_enabled", False)
                            ),
                            "bayes_band_force": bool(config.get("bayes_band_force", False)),
                            "bayes_band_max_draws": int(
                                config.get("bayes_band_max_draws", 512) or 512
                            ),
                            "bayes_diag_ess_min": float(
                                config.get("bayes_diag_ess_min", 200.0)
                            ),
                            "bayes_diag_rhat_max": float(
                                config.get("bayes_diag_rhat_max", 1.05)
                            ),
                            "bayes_diag_mcse_mean": float(
                                config.get("bayes_diag_mcse_mean", float("inf"))
                            ),
                            # Router expects a non-negative int; 0 means "auto".
                            "bayes_walkers": walkers_cfg,
                            "bayes_burn": burn_cfg,
                            "bayes_steps": steps_cfg,
                            "bayes_thin": thin_cfg,
                            "bayes_prior_sigma": str(
                                config.get("bayes_prior_sigma", "half_cauchy")
                            ),
                            "abort_event": abort_event,
                        })
                    seed_val = perf_seed if perf_seed_all and (perf_seed is not None) else None
                    workers_eff = int(unc_workers)
                    strategy = str(config.get("perf_parallel_strategy", "outer"))
                    try:
                        _bt_raw = int(config.get("perf_blas_threads", 0) or 0)
                    except Exception:
                        _bt_raw = 0
                    blas_effective = 1 if strategy == "outer" else (_bt_raw if _bt_raw > 0 else "lib")

                    perf_line = (
                        f"[DEBUG] perf: fit_workers={cfg_perf.fit_workers} unc_workers={workers_eff} "
                        f"blas_threads={blas_effective} seed_all={perf_seed_all} seed={seed_val}"
                    )
                    logger.debug(perf_line)
                    if log:
                        log(perf_line)

                    with _tp_limits_ctx_for_config(config):
                        performance.apply_global_seed(seed_val, perf_seed_all)
                        unc_res = route_uncertainty(
                            unc_method_canon,
                            theta_hat=theta_hat,
                            residual_fn=residual_fn,
                            jacobian=jac,
                            model_eval=predict_fit_fn,
                            fit_ctx=fit_ctx,
                            x_all=x_all,
                            y_all=y_all,
                            workers=(workers_eff if workers_eff > 0 else None),
                            seed=seed_val,
                            n_boot=(n_boot if n_boot is not None else 0),
                        )
            except Exception as exc:
                msg = str(exc)
                if "Unknown uncertainty method" in msg:
                    raise
                err_msg = f"{Path(path).name}: uncertainty failed: {msg}"
                logger.warning(err_msg)
                if log:
                    log(err_msg)
                uncertainty_failures.append((Path(path).name, msg))
                if abort_on_uncertainty_failure:
                    raise
                unc_error_msg = msg
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

            base_path = out_dir / stem
            band = unc_norm.get("band") or unc_norm.get("prediction_band")
            is_bootstrap = method_lbl.lower().startswith("bootstrap")

            if is_bootstrap:
                data_io.write_uncertainty_csvs(
                    base_path,
                    path,
                    unc_norm,
                    write_wide=True,
                )
                txt_path = base_path.with_name(base_path.name + "_uncertainty.txt")
                solver_meta = res.get("solver_meta", {}) if isinstance(res, dict) else {}
                baseline_meta = res.get("baseline_meta", {}) if isinstance(res, dict) else {}
                perf_meta = res.get("perf_meta", {}) if isinstance(res, dict) else {}
                locks = res.get("locks", []) if isinstance(res, dict) else []
                try:
                    data_io._write_unc_txt(txt_path, path, unc_norm, solver_meta, baseline_meta, perf_meta, locks)
                except Exception:
                    data_io.write_uncertainty_txt(
                        txt_path,
                        unc_norm,
                        peaks=fitted,
                        method_label=method_lbl,
                        file_path=path,
                        solver_meta=solver_meta,
                        baseline_meta=baseline_meta,
                        perf_meta=perf_meta,
                        locks=locks,
                    )

                if band_pref_bootstrap and band is not None:
                    if isinstance(band, (tuple, list)) and len(band) == 3:
                        xb, lob, hib = band
                        band_csv = base_path.with_name(base_path.name + "_band.csv")
                        with band_csv.open("w", newline="", encoding="utf-8") as fh:
                            writer = csv.writer(fh, lineterminator="\n")
                            writer.writerow(["x", "ci_lo", "ci_hi"])
                            for xi, lo, hi in zip(xb, lob, hib):
                                if abort_event is not None and getattr(abort_event, "is_set", lambda: False)():
                                    return {"aborted": True, "records": [], "reason": "user-abort"}
                                try:
                                    xi_f = float(xi)
                                    lo_f = float(lo)
                                    hi_f = float(hi)
                                except Exception:
                                    continue
                                if not (math.isfinite(xi_f) and math.isfinite(lo_f) and math.isfinite(hi_f)):
                                    continue
                                writer.writerow([xi_f, lo_f, hi_f])
            else:
                data_io.write_uncertainty_txt(
                    out_dir / f"{stem}_uncertainty.txt",
                    unc_norm,
                    peaks=fitted,
                    method_label=method_lbl,
                    file_path=path,
                )

                data_io.write_uncertainty_csvs(
                    base_path,
                    path,
                    unc_norm,
                    write_wide=export_unc_wide,
                )

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

        if expected_unc and unc_res is None:
            if unc_error_msg is None:
                msg = "Batch export requested uncertainty but none was computed"
                logger.error(msg)
                if log:
                    log(f"{Path(path).name}: {msg}")
                raise RuntimeError(msg)

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

    if uncertainty_failures:
        details = "; ".join(f"{name}: {msg}" for name, msg in uncertainty_failures[:5])
        if len(uncertainty_failures) > 5:
            details += f"; +{len(uncertainty_failures) - 5} more"
        summary_msg = f"Uncertainty failed for {len(uncertainty_failures)} file(s)"
        full_msg = f"{summary_msg}: {details}" if details else summary_msg
        logger.warning(full_msg)
        if log:
            log(full_msg)

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
