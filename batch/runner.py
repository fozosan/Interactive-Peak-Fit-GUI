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

if os.environ.get("SMOKE_MODE") == "1":  # pragma: no cover - environment safeguard
    os.environ.setdefault("MPLBACKEND", "Agg")

import core.data_io as data_io
from core import models, peaks, signals, fit_api, baseline_model, pipeline
from core.residuals import build_residual
from infra import performance


logger = logging.getLogger(__name__)

# The per-file status table, <output_base>_files.csv, has one row per matched
# file. Its status is one of:
#   ok                  fitted, and the solver converged
#   not_converged       the solver stopped without converging
#   height_at_cap       a height ended at the cap the window's data allow (named in
#                       message, fit.bounds.height_cap): the fit put more into
#                       that peak than the data hold; fit_ok False, no uncertainty
#   fit_error           the baseline or the fit raised an error
#   skipped_format      not two-column numeric data
#   skipped_unreadable  could not be opened (permissions, or a directory)
#   skipped_no_points   the fit window has no more points than free parameters
#   no_peaks            no peaks to fit, e.g. auto seeding found none
#   not_processed       the batch stopped before this file
# Its unc_status, with uncertainty on, is ok, failed, skipped (no usable fit),
# or aborted (the user aborted the run during this file's uncertainty: no result).
# Its flags are advice, never a veto (";"-separated):
#   bound_hit           a fitted parameter sits on its bound (named in message)
#   ci_at_bound         an uncertainty interval reaches a parameter bound
#   mcmc_short_chain    the Bayesian chain is under 50 integrated autocorrelation
#                       times long (emcee's rule); unc_message gives its length
# The noise_* columns record the spectrum's noise model, var_i = s0^2 + g * S_i
# with S_i the fitted signal above the baseline, which weighted the fit: s0, g,
# the n points it was estimated from and how (fit_residuals_d4: from the 4th
# differences of a first fit's residuals, core.noise_model). Blank when no model
# was estimated (noise_model "none").
# baseline_fit says how the fit treated the baseline: "joint", fitted with the
# peaks as Legendre columns of order baseline_order over the fit window (the
# batch default, decision D2), "fixed", held at the ALS or polynomial baseline
# (baseline_order blank), or "none" with use_baseline off. Blank when no fit ran.
STATUS_COLUMNS = (
    "file",
    "path",
    "status",
    "flags",
    "message",
    "n_points_window",
    "n_peaks",
    "rmse",
    "fit_ok",
    "baseline_fit",
    "baseline_order",
    "noise_s0",
    "noise_g",
    "noise_n",
    "noise_method",
    "unc_method",
    "unc_status",
    "unc_message",
)


def _one_line(text) -> str:
    return " ".join(str(text).split())


def _fit_failure_status(exc: Exception) -> str:
    """Status for a file whose baseline or fit raised ``exc``."""

    # The Classic solver runs scipy.optimize.curve_fit, which reports a fit that
    # stops without converging by raising this; the other solvers return
    # fit_ok=False instead.
    if isinstance(exc, RuntimeError) and str(exc).startswith("Optimal parameters not found"):
        return "not_converged"
    return "fit_error"


def _seed_peaks(config: dict, solver_name: str) -> List[peaks.Peak]:
    """Return the seed peaks in ``config["peaks"]``.

    The GUI sends its own peak dicts for Current and Template seeds. Their FWHM
    caps reach the solvers as ``width_caps``: when the options for
    ``solver_name`` hold no cap, the peaks' own caps are written there.
    """

    records = list(config.get("peaks") or [])
    caps = peaks.width_caps(records)
    opts = dict(config.get(solver_name) or {})
    if any(c is not None for c in caps) and not any(
        c is not None for c in (opts.get("width_caps") or [])
    ):
        opts["width_caps"] = caps
        config[solver_name] = opts
    return peaks.deserialize(records)


def _write_status_table(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


_tp_limits_ctx_for_config = pipeline.threadpool_limits_for
_file_seed = pipeline.file_seed


_norm_jitter = pipeline.norm_jitter


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

    A file that cannot be read or fitted costs only that file. Every matched
    file gets a row in ``<output_base>_files.csv`` (see ``STATUS_COLUMNS``),
    which is written together with the peak summary even when the batch is
    aborted or stopped by an error.

    Returns ``(ok, n)``: the files whose fit converged and the files matched.
    An abort returns ``{"aborted": True, ...}`` instead.
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

    solver_name = config.get("solver", "classic")
    # If a generic "solver_options" block is provided, merge into the
    # per-solver config so run_fit_consistent → pack_theta_bounds sees caps.
    if "solver_options" in config:
        cfg_opts = dict(config.get(solver_name, {}))
        cfg_opts.update(dict(config.get("solver_options") or {}))
        config[solver_name] = cfg_opts
    base_template = _seed_peaks(config, solver_name)
    # Batch fits weight every spectrum by its own noise model unless the config
    # says noise_model "none" (decision D1). The GUI's core default is "none".
    fit_config = dict(config)
    fit_config.setdefault("noise_model", "per_spectrum")
    fit_api._noise_model_choice(fit_config)  # a wrong setting stops the batch here
    # The batch fits the baseline with the peaks unless the config says
    # baseline_fit "fixed" (decision D2); the GUI's core default is "fixed".
    fit_config.setdefault("baseline_fit", "joint")
    # Use baseline off (the GUI's checkbox) means no baseline at all: neither a
    # computed one nor one fitted with the peaks.
    use_baseline = bool(config.get("use_baseline", True))
    if not use_baseline:
        fit_config["baseline_fit"] = "fixed"
    baseline_model.baseline_fit_choice(fit_config)  # a wrong setting stops the batch here
    baseline_model.baseline_order(fit_config)
    mode = config.get("mode", "add")
    base_cfg = config.get("baseline", {})
    save_traces = bool(config.get("save_traces", False))
    source = config.get("source", "template")
    reheight = bool(config.get("reheight", False))
    auto_max = int(config.get("auto_max", 5))
    cfg_perf = performance.get_parallel_config()
    performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)

    out_dir = Path(
        config.get("output_dir", Path(config.get("peak_output", "peaks.csv")).parent)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.get("output_base")
    if base_name is None:
        peak_output_cfg = config.get("peak_output", out_dir / "batch_fit.csv")
        base_name = Path(peak_output_cfg).stem.replace("_fit", "")
    peak_output = out_dir / f"{base_name}_fit.csv"
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
    compute_uncertainty = compute_uncertainty_flag
    # Choose uncertainty method: explicit arg > config aliases > default
    unc_choice = (
        unc_method_arg
        or _method_key(config.get("unc_method"))
        or _method_key(config.get("uncertainty_method"))
        or _method_key(unc_section)
        or "asymptotic"
    )
    unc_method_canon = data_io.canonical_unc_label(unc_choice)
    if compute_uncertainty and unc_method_canon == "unknown":
        # A setting is wrong, not a file: stop before any file is fitted.
        raise ValueError(f"Unknown uncertainty method: {unc_choice!r}")
    if str(unc_method_canon).lower() == "bootstrap":
        pipeline.bootstrap_draws(config)  # a missing or bad bootstrap_n stops the batch here
    perf_seed_all, perf_seed = pipeline.seed_settings(config)
    if log:
        log(f"batch uncertainty method={unc_method_canon}")
        log(f"batch compute_uncertainty={bool(compute_uncertainty)}")

    records = []
    unc_rows = []
    uncertainty_failures = []
    status_rows = [
        {
            **dict.fromkeys(STATUS_COLUMNS, ""),
            "file": Path(path).name,
            "path": str(path),
            "status": "not_processed",
        }
        for path in files
    ]
    status_path = out_dir / f"{base_name}_files.csv"
    inputs = {os.path.normcase(os.path.abspath(path)) for path in files}
    stems = pipeline.output_stems(files)

    def _process_file(path: str, stem: str, entry: dict) -> None:
        """Fit one file and write its outputs under ``stem``, recording the outcome in ``entry``."""

        try:
            x, y = data_io.load_xy(path)
        except OSError as exc:
            # Unreadable, or a directory whose name matched a pattern.
            entry.update(status="skipped_unreadable", message=_one_line(exc))
            if log:
                log(f"{Path(path).name}: skipped (unreadable: {entry['message']})")
            return
        except ValueError as exc:
            emsg = str(exc).lower()
            if (
                "more than 2" in emsg
                or "more than two" in emsg
                or "unsupported column" in emsg
            ):
                reason = "has >2 columns; mapping files not supported"
            elif "could not parse" in emsg:
                reason = "no two-column numeric data"
            else:
                raise
            entry.update(status="skipped_format", message=reason)
            if log:
                log(f"{Path(path).name}: skipped ({reason})")
            return
        xmin = config.get("fit_xmin")
        xmax = config.get("fit_xmax")
        if xmin is not None and xmax is not None:
            lo, hi = sorted((float(xmin), float(xmax)))
            mask = (x >= lo) & (x <= hi)
        else:
            mask = np.ones_like(x, bool)
        entry["n_points_window"] = int(np.count_nonzero(mask))

        method = str(base_cfg.get("method", "als")).lower()
        use_slice = bool(config.get("baseline_uses_fit_range", True))
        mask_eff = mask if use_slice else None
        if not use_baseline:
            baseline = np.zeros_like(y, dtype=float)
        elif method == "als":
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
        entry["n_peaks"] = len(template)

        peaks_in = copy.deepcopy(template)
        seed = _file_seed(path, perf_seed) if perf_seed_all else None
        res = fit_api.run_fit_consistent(
            x,
            y,
            peaks_in,
            fit_config,
            baseline,
            mode,
            mask,
            reheight=reheight,
            rng_seed=seed,
            verbose=bool(log),
            return_jacobian=True,
            return_predictors=True,
        )
        if res.get("skip_reason") is not None:
            entry.update(
                status="no_peaks" if res["skip_reason"] == "no_peaks" else "skipped_no_points",
                message=res.get("message", ""),
            )
            if log:
                log(f"{Path(path).name}: skipped ({entry['message']})")
            return

        fitted = res["peaks_out"]
        theta = np.asarray(res["theta"], dtype=float)  # noqa: F841 - for debugging
        rmse = float(res["rmse"])
        # The baseline the fit used: the one computed above, or the one it fitted.
        if res.get("baseline") is not None:
            baseline = np.asarray(res["baseline"], float)
        mode_used = res.get("mode", mode)
        joint_fit = res.get("joint_baseline") is not None
        entry.update(
            baseline_fit=pipeline.baseline_fit_label(res, config),
            baseline_order=res.get("baseline_order") if joint_fit else "",
        )
        noise_info = res.get("noise_model")
        if noise_info:
            entry.update(
                noise_s0=float(noise_info["s0"]),
                noise_g=float(noise_info["g"]),
                noise_n=int(noise_info["n"]),
                noise_method=str(noise_info["method"]),
            )
        if os.environ.get("IPF_SHADOW") == "1":
            x_fit = x[mask]
            y_fit = (y if mode_used == "add" else y - baseline)[mask]
            base_fit = baseline[mask] if mode_used == "add" else None
            resid_fn = build_residual(
                x_fit, y_fit, fitted, mode_used, base_fit, "linear", None
            )
            r = resid_fn(theta)
            rmse_shadow = float(np.sqrt(np.mean(r * r))) if r.size else float("nan")
            if abs(rmse_shadow - rmse) > 1e-8 and log:
                log(f"shadow rmse diff {rmse_shadow - rmse}")
        # One row builder and one writer for the GUI's Export and batch (2.1). An
        # earlier run's uncertainty files are dealt with once this file's own
        # uncertainty is written, below.
        base_path = out_dir / stem
        local_records = pipeline.peak_records(Path(path).name, res, config, x)
        pipeline.write_spectrum(
            base_path, path, x=x, y_raw=y, fit=res, settings=config, trace=save_traces, keep=inputs, log=log,
            retire_stale=False,
        )
        trace_path = base_path.with_name(base_path.name + "_trace.csv") if save_traces else None
        records.extend(local_records)
        notes = [] if res["fit_ok"] else [res.get("message") or "the solver did not converge"]
        if res.get("bound_hit"):
            notes.append("at bound: " + ", ".join(res["bound_hit"]))
        entry.update(
            status="ok" if res["fit_ok"] else "height_at_cap" if res.get("height_at_cap") else "not_converged",
            flags=";".join(res.get("flags") or []),
            message=_one_line("; ".join(notes)),
            n_peaks=len(fitted),
            rmse=rmse,
            fit_ok=bool(res["fit_ok"]),
        )

        # Uncertainty problems stay with this file: they are recorded in its
        # status row and never stop the batch.
        unc_res = None
        unc_written = []
        if res["fit_ok"] and fitted and compute_uncertainty:
            try:
                # The one uncertainty context, shared with the GUI (game plan 2.1).
                unc_res = pipeline.compute_uncertainty(
                    res, x, y, mask, config, unc_method_canon,
                    fit_config=fit_config, file_seed=seed, abort_event=abort_event, log=log,
                )
                if unc_res is None:
                    raise RuntimeError("the uncertainty method returned no result")
                # The writers' form, with the fit's noise model, baseline, rmse and
                # dof; one set of files for every method, as the GUI writes them.
                unc_norm = pipeline.finish_uncertainty(unc_res, res, config, unc_method_canon)
                method_lbl = unc_norm["label"]
                unc_written = pipeline.write_uncertainty(base_path, path, unc_norm, res, config)

                file_rows = list(data_io.iter_uncertainty_rows(path, unc_norm))
                unc_rows.extend(file_rows)
                if log:
                    log(f"{Path(path).name}: uncertainty={method_lbl}")
                entry["unc_status"] = "ok"
                if any(r.get("flags") for r in file_rows):
                    entry["flags"] = ";".join(f for f in (entry["flags"], "ci_at_bound") if f)
                unc_diag = (
                    unc_res.get("diagnostics") if isinstance(unc_res, dict) else getattr(unc_res, "diagnostics", None)
                ) or {}
                n_fail = int(unc_diag.get("n_fail") or 0)
                if n_fail:
                    entry["unc_message"] = (
                        f"{n_fail} of {int(unc_diag.get('n_boot') or 0)} bootstrap refits failed and were left out"
                    )
                if unc_diag.get("chain_short"):
                    entry["flags"] = ";".join(f for f in (entry["flags"], "mcmc_short_chain") if f)
                    ratio = unc_diag.get("chain_tau_ratio")
                    entry["unc_message"] = (
                        f"MCMC chain is {float(ratio):.1f} autocorrelation times long; emcee advises at least "
                        f"{float(unc_diag.get('chain_tau_min') or 50):g}: run more bayes_steps"
                        if ratio is not None and math.isfinite(float(ratio))
                        else "the MCMC autocorrelation time could not be estimated: run more bayes_steps"
                    )
            except pipeline.UncertaintyAborted as exc:
                # No result, and none written: the batch stops before the next file.
                entry.update(unc_status="aborted", unc_message=_one_line(str(exc)))
                if log:
                    log(f"{Path(path).name}: uncertainty aborted")
            except Exception as exc:
                msg = str(exc)
                err_msg = f"{Path(path).name}: uncertainty failed: {msg}"
                logger.warning(err_msg)
                if log:
                    log(err_msg)
                uncertainty_failures.append((Path(path).name, msg))
                entry.update(unc_status="failed", unc_message=_one_line(msg))
                # write_uncertainty took back any file it had half written.
        # An earlier run's uncertainty files that this one did not replace: with
        # uncertainty on, those the app wrote go; with it off, nothing is deleted.
        pipeline.retire_stale_uncertainty(base_path, unc_written, keep=inputs, log=log, remove=bool(compute_uncertainty))

        if log:
            msg = f"{Path(path).name}: {'ok' if res['fit_ok'] else 'fail'} rmse={rmse:.3g}"
            if trace_path:
                msg += f" {trace_path}"
            log(msg)

    ok = 0
    aborted = False
    try:
        for i, (path, stem, entry) in enumerate(zip(files, stems, status_rows), 1):
            if abort_event is not None and getattr(abort_event, "is_set", lambda: False)():
                aborted = True
                break
            if progress:
                progress(i, total, path)
            try:
                _process_file(path, stem, entry)
            except Exception as exc:
                entry.update(
                    status=_fit_failure_status(exc),
                    message=_one_line(f"{type(exc).__name__}: {exc}"),
                    rmse="",
                    fit_ok=False,
                )
                err_msg = f"{Path(path).name}: {entry['status']}: {entry['message']}"
                logger.warning(err_msg)
                if log:
                    log(err_msg)
            if compute_uncertainty:
                entry["unc_method"] = unc_method_canon
                if not entry["unc_status"]:
                    entry["unc_status"] = "skipped"
                    entry["unc_message"] = {
                        "not_converged": "the fit did not converge",
                        "height_at_cap": "a height ended at its cap",
                    }.get(entry["status"], "no fit")
            if entry["status"] == "ok":
                ok += 1
            if abort_on_uncertainty_failure and entry["unc_status"] == "failed":
                raise RuntimeError(
                    f"{entry['file']}: uncertainty failed: {entry['unc_message']}"
                )
    finally:
        # Written whatever happened above: an abort or an error leaves a
        # partial summary and a status row for every matched file.
        _write_status_table(status_path, status_rows)
        peak_csv = data_io.build_peak_table(records)
        with peak_output.open("w", encoding="utf-8", newline="") as fh:
            fh.write(peak_csv)
        if unc_rows:
            data_io.write_batch_uncertainty_long(out_dir, unc_rows)
    if log:
        log(f"file status table: {status_path}")

    if uncertainty_failures:
        details = "; ".join(f"{name}: {msg}" for name, msg in uncertainty_failures[:5])
        if len(uncertainty_failures) > 5:
            details += f"; +{len(uncertainty_failures) - 5} more"
        summary_msg = f"Uncertainty failed for {len(uncertainty_failures)} file(s)"
        full_msg = f"{summary_msg}: {details}" if details else summary_msg
        logger.warning(full_msg)
        if log:
            log(full_msg)

    if aborted:
        return {"aborted": True, "records": [], "reason": "user-abort"}
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
