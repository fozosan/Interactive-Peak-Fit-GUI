"""One spectrum's outputs, built and written the same way by the GUI and by batch.

Game plan 2.1 and 2.2. The GUI's Export and batch used to build their own
``_fit.csv`` rows and write their own uncertainty files, and 9 of the 43 fit
columns came out different. Both now hand a fit result
(``core.fit_api.run_fit_consistent``) and the settings it ran with to this
module:

- :func:`peak_records` is the one row builder of ``<base>_fit.csv``;
- :func:`write_spectrum` writes a spectrum's fit table, its trace and its
  uncertainty files, and removes the uncertainty files of an earlier fit that
  this one does not write;
- :func:`finish_uncertainty` turns an uncertainty method's result into what the
  writers and batch's summary take;
- :func:`output_stems` names each batch input's outputs, keeping inputs that
  share a stem apart;
- :func:`unfitted` describes peaks no fit produced, for an export before a fit.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import os
import zlib
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from infra import performance

from . import baseline_model, data_io, fit_api, models
from .peaks import Peak
from .residuals import build_residual, finite_diff_jacobian
from .uncertainty import UncertaintyResult, bootstrap_ci
from .uncertainty_router import route_uncertainty

logger = logging.getLogger(__name__)

# The uncertainty files a spectrum can have, after its output stem. "_band.csv"
# is the name batch gave the bootstrap band before every method's band became
# "_uncertainty_band.csv"; an old one is removed with the rest.
UNCERTAINTY_SUFFIXES = (
    "_uncertainty.csv",
    "_uncertainty_wide.csv",
    "_uncertainty.txt",
    "_uncertainty_band.csv",
    "_band.csv",
)


def file_seed(path, base_seed: Optional[int]) -> int:
    """Seed for one file's fit restarts: the same on every run and machine.

    CRC-32 of the file name, started from the base seed. Python's ``hash()``,
    used before, is salted per process, so the seeds changed on every run.
    """

    return zlib.crc32(Path(path).name.encode("utf-8"), int(base_seed or 0) & 0xFFFFFFFF)


def output_stems(paths: Sequence[str]) -> List[str]:
    """The stem of each input's outputs, in the order of ``paths``.

    An input's own stem, unless another input shares it (``a.csv`` and ``a.txt``,
    compared without case, as Windows does): then ``<stem>_<extension>``, and a
    counter after that if names still clash. Without this the inputs overwrote
    each other's outputs.
    """

    stems = [Path(p).stem for p in paths]
    shared = Counter(s.casefold() for s in stems)
    names = [
        f"{s}_{Path(p).suffix.lstrip('.')}" if shared[s.casefold()] > 1 and Path(p).suffix else s
        for p, s in zip(paths, stems)
    ]
    seen: Counter = Counter()
    out = []
    for name in names:
        seen[name.casefold()] += 1
        out.append(name if seen[name.casefold()] == 1 else f"{name}_{seen[name.casefold()]}")
    return out


def _locked_mask(peaks: Sequence[Any]) -> np.ndarray:
    # No solver fits eta: it is locked with the locked centres and widths.
    return np.array([f for p in peaks for f in (p.lock_center, False, p.lock_width, True)], bool)


def unfitted(x, y, peaks_in: Sequence[Any], baseline, mode: str, mask) -> dict:
    """A fit-shaped record of peaks that no fit produced.

    For an export of peaks placed by hand, from a template, or edited after the
    last fit: ``fit_ok`` is False, no solver set bounds, and the rmse is that of
    these peaks over the window.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    out_peaks = [Peak(p.center, p.height, p.fwhm, p.eta, bool(p.lock_center), bool(p.lock_width)) for p in peaks_in]
    theta = np.array([v for p in out_peaks for v in (p.center, p.height, p.fwhm, p.eta)], float)
    base_fit = None if baseline is None else np.asarray(baseline, float)[mask]
    resid = build_residual(x[mask], y[mask], out_peaks, mode, base_fit, "linear", None)
    r = resid(theta) if theta.size else np.zeros(0)
    locked = _locked_mask(out_peaks)
    n_free = int(np.count_nonzero(~locked))
    return {
        "peaks_out": out_peaks,
        "theta": theta,
        "rmse": float(np.sqrt(np.mean(r * r))) if r.size else float("nan"),
        "fit_ok": False,
        "message": "not fitted",
        "bounds": None,
        "mode": mode,
        "baseline": None if baseline is None else np.asarray(baseline, float),
        "baseline_fit": "fixed",
        "baseline_order": None,
        "joint_baseline": None,
        "noise_model": None,
        "noise_sigma": None,
        "locked_mask": locked,
        "param_names": [f"{n}{i + 1}" for i in range(len(out_peaks)) for n in ("center", "height", "fwhm", "eta")],
        "dof": r.size - n_free if r.size > n_free else float("nan"),
        "restarts_used": float("nan"),
        "jitter_pct": float("nan"),
        "solver_options": {},
    }


def peak_records(file_name: str, fit: Mapping[str, Any], settings: Mapping[str, Any], x) -> List[dict]:
    """The rows of ``<base>_fit.csv`` for one fit, one per peak.

    ``fit`` is ``run_fit_consistent``'s result (or :func:`unfitted`) and
    ``settings`` the settings it ran with. Everything a row says about the fit
    comes from the fit: ``fit_ok``, the solver options it ran with, its
    restarts, and the bounds it held the parameters to (blank when no fit ran).
    """

    fitted = list(fit.get("peaks_out") or [])
    areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in fitted]
    total = sum(areas) or 1.0
    opts = dict(fit.get("solver_options") or {})
    bounds = fit.get("bounds")
    lo, hi = (None, None) if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
    x = np.asarray(x, float)
    base_cfg = dict(settings.get("baseline") or {})
    method = str(base_cfg.get("method", "als")).lower()
    joint = fit.get("baseline_fit") == "joint"
    noise_weighted = bool(fit.get("noise_model")) and fit.get("noise_sigma") is not None
    window = (settings.get("fit_xmin"), settings.get("fit_xmax"))
    fit_lo, fit_hi = (
        (float(window[0]), float(window[1]))
        if None not in window
        else (float(np.min(x)), float(np.max(x)))
    )

    def bound(arr, i, k):
        return float(arr[4 * i + k]) if arr is not None else np.nan

    rows = []
    for i, (p, area) in enumerate(zip(fitted, areas)):
        rows.append(
            {
                "file": file_name,
                "peak": i + 1,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "lock_width": p.lock_width,
                "lock_center": p.lock_center,
                "area": area,
                "area_pct": 100.0 * area / total,
                "rmse": fit.get("rmse", np.nan),
                "fit_ok": bool(fit.get("fit_ok", False)),
                "mode": fit.get("mode"),
                "fit_xmin": fit_lo,
                "fit_xmax": fit_hi,
                "solver_choice": settings.get("solver"),
                "solver_loss": opts.get("loss", np.nan),
                "solver_weight": "noise_model" if noise_weighted else opts.get("weights", np.nan),
                "solver_fscale": opts.get("f_scale", np.nan),
                "solver_maxfev": opts.get("maxfev", np.nan),
                "solver_restarts": fit.get("restarts_used", np.nan),
                "solver_jitter_pct": fit.get("jitter_pct", np.nan),
                "use_baseline": bool(settings.get("use_baseline", True)),
                "baseline_mode": fit.get("mode"),
                "baseline_uses_fit_range": bool(settings.get("baseline_uses_fit_range", True)),
                "perf_numba": bool(settings.get("perf_numba", False)),
                "perf_gpu": bool(settings.get("perf_gpu", False)),
                "perf_cache_baseline": bool(settings.get("perf_cache_baseline", True)),
                "perf_seed_all": bool(settings.get("perf_seed_all", False)),
                "perf_max_workers": int(settings.get("perf_max_workers", 0) or 0),
                "bounds_center_lo": bound(lo, i, 0),
                "bounds_center_hi": bound(hi, i, 0),
                "bounds_fwhm_lo": bound(lo, i, 2),
                "bounds_height_lo": bound(lo, i, 1),
                "bounds_height_hi": bound(hi, i, 1),
                "x_scale": np.nan,
                "baseline_fit": fit.get("baseline_fit", ""),
                "baseline_order": fit.get("baseline_order") if joint else "",
                "baseline_method": method,
                **(
                    {
                        "als_lam": base_cfg.get("lam"),
                        "als_p": base_cfg.get("p"),
                        "als_niter": base_cfg.get("niter"),
                        "als_thresh": base_cfg.get("thresh"),
                    }
                    if method == "als"
                    else dict.fromkeys(("als_lam", "als_p", "als_niter", "als_thresh"), np.nan)
                ),
                **(
                    {
                        "poly_degree": int(base_cfg.get("degree", 2)),
                        "poly_normalize_x": bool(base_cfg.get("normalize_x", True)),
                    }
                    if method == "polynomial"
                    else {"poly_degree": np.nan, "poly_normalize_x": np.nan}
                ),
            }
        )
    return rows


def locks(fit: Mapping[str, Any]) -> List[dict]:
    """Per-peak locks, in the order the uncertainty TXT reports parameters."""

    lm = fit.get("locked_mask")
    if lm is None:
        return []
    lm = np.asarray(lm, bool)
    return [dict(zip(("center", "height", "fwhm", "eta"), map(bool, lm[4 * i : 4 * i + 4]))) for i in range(lm.size // 4)]


def _baseline_record(fit: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    """How the fit treated the baseline, as the uncertainty writers record it."""

    jb = fit.get("joint_baseline")
    if jb is None:
        return {"fit": "fixed", "method": str((settings.get("baseline") or {}).get("method", "als")).lower()}
    out = jb.as_dict()
    out["coef"] = [float(v) for v in np.asarray(fit["baseline_coef"], float)]
    return out


def _pop_baseline_stats(unc_res, n_peak_params: int, names: Sequence[str]) -> dict:
    """Take the joint baseline's coefficients out of a method's statistics.

    The asymptotic method keys them ``p<i>`` (i after the peaks'), the bootstrap
    and the Bayesian by name (``baseline_c<j>``). Returns their estimate, standard
    error and interval, one entry per coefficient, NaN where a method has none.
    """

    stats = unc_res.get("stats") if isinstance(unc_res, dict) else getattr(unc_res, "stats", None)
    out = {"coef": [], "sd": [], "ci_lo": [], "ci_hi": []}
    for j, name in enumerate(names):
        blk = None
        for key in (name, f"p{int(n_peak_params) + j}"):
            got = stats.pop(key, None) if isinstance(stats, dict) else None
            if blk is None and isinstance(got, dict):
                blk = got
        blk = blk or {}

        def _num(*keys):
            for k in keys:
                if blk.get(k) is not None:
                    try:
                        return float(blk[k])
                    except (TypeError, ValueError):
                        continue
            return float("nan")

        out["coef"].append(_num("est"))
        out["sd"].append(_num("sd"))
        out["ci_lo"].append(_num("p2_5", "p2.5", "ci_lo"))
        out["ci_hi"].append(_num("p97_5", "p97.5", "ci_hi"))
    return out


def finish_uncertainty(unc_res: Any, fit: Mapping[str, Any], settings: Mapping[str, Any], method: str) -> dict:
    """An uncertainty method's result, as the writers and batch's summary take it.

    Its label is canonical; the fit's noise model, baseline, rmse and degrees of
    freedom go with it. A joint baseline's coefficients are reported with the
    baseline, never as another peak's parameters. Missing statistics stay NaN,
    which the writers leave blank: a NaN rmse is not 0, and dof <= 0 is not 1.
    """

    baseline_info = _baseline_record(fit, settings)
    if fit.get("joint_baseline") is not None:
        baseline_info.update(
            _pop_baseline_stats(unc_res, int(np.asarray(fit["theta"]).size), fit["joint_baseline"].names())
        )
    unc_norm = data_io.normalize_unc_result(unc_res)
    unc_norm["label"] = data_io.canonical_unc_label(unc_norm.get("label") or method)
    unc_norm["noise_model"] = fit.get("noise_model")
    unc_norm["baseline_model"] = baseline_info
    unc_norm["rmse"] = float(fit.get("rmse", float("nan")))
    # The fit's degrees of freedom (points less the parameters it fitted, eta
    # never among them); the method's own only when the fit has none.
    dof = float("nan")
    for source in (fit.get("dof"), unc_norm.get("dof")):
        try:
            dof = float(source)
        except (TypeError, ValueError):
            continue
        if math.isfinite(dof):
            break
    unc_norm["dof"] = int(dof) if math.isfinite(dof) and dof > 0 else float("nan")
    return unc_norm


def remove_uncertainty_files(base, keep: Iterable[str] = (), log=None) -> None:
    """Delete the uncertainty files under ``base``, which an earlier fit left.

    ``keep`` holds paths never to touch (a batch's inputs), normalised with
    ``os.path.normcase(os.path.abspath(...))``.
    """

    base = Path(base)
    keep = set(keep)
    for suffix in UNCERTAINTY_SUFFIXES:
        old = base.with_name(base.name + suffix)
        if os.path.normcase(os.path.abspath(old)) in keep:
            continue
        try:
            old.unlink(missing_ok=True)
        except OSError as exc:
            msg = f"could not remove stale {old.name}: {exc}"
            logger.warning(msg)
            if log:
                log(msg)


def _band(unc_norm: Mapping[str, Any]):
    band = unc_norm.get("band")
    if band is None:
        band = unc_norm.get("prediction_band")
    if isinstance(band, (tuple, list)) and len(band) == 3:
        return band
    return None


def write_uncertainty(base, file_path, unc_norm: Mapping[str, Any], fit: Mapping[str, Any],
                      settings: Mapping[str, Any]) -> List[Path]:
    """Write ``<base>_uncertainty.csv`` and ``.txt``, the wide CSV when
    ``settings["export_unc_wide"]``, and ``<base>_uncertainty_band.csv`` when the
    method gave a band: one set of files for every method and both paths."""

    base = Path(base)
    long_csv, wide_csv = data_io.write_uncertainty_csvs(
        base, file_path, unc_norm, write_wide=bool(settings.get("export_unc_wide", False))
    )
    written = [Path(long_csv)] + ([Path(wide_csv)] if wide_csv else [])
    txt = base.with_name(base.name + "_uncertainty.txt")
    data_io.write_uncertainty_txt(
        txt,
        unc_norm,
        peaks=fit.get("peaks_out"),
        method_label=unc_norm.get("label", ""),
        file_path=file_path,
        locks=locks(fit),
    )
    written.append(txt)
    band = _band(unc_norm)
    if band is not None:
        band_csv = base.with_name(base.name + "_uncertainty_band.csv")
        level = data_io.ci_level_text(unc_norm.get("alpha", 0.05))
        with band_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["x", f"y_lo{level}", f"y_hi{level}"])
            for xi, lo, hi in zip(*band):
                xi, lo, hi = float(xi), float(lo), float(hi)
                if math.isfinite(xi) and math.isfinite(lo) and math.isfinite(hi):
                    writer.writerow([xi, lo, hi])
        written.append(band_csv)
    return written


def write_spectrum(
    base,
    file_path,
    *,
    x,
    y_raw,
    fit: Mapping[str, Any],
    settings: Mapping[str, Any],
    unc_norm: Optional[Mapping[str, Any]] = None,
    trace: bool = True,
    keep: Iterable[str] = (),
    log=None,
) -> List[Path]:
    """Write one spectrum's files under ``base`` (the path the suffixes follow).

    ``<base>_fit.csv`` from :func:`peak_records`; ``<base>_trace.csv`` when
    ``trace``, with the baseline the fit used; and the uncertainty files of
    ``unc_norm`` (from :func:`finish_uncertainty`). The uncertainty files of an
    earlier fit go first, so the folder never pairs this fit with another
    fit's uncertainty, even when ``unc_norm`` is None. Returns the paths
    written.
    """

    base = Path(base)
    fit_csv = base.with_name(base.name + "_fit.csv")
    with fit_csv.open("w", encoding="utf-8", newline="") as fh:
        fh.write(data_io.build_peak_table(peak_records(Path(file_path).name, fit, settings, x)))
    written = [fit_csv]
    if trace:
        trace_csv = base.with_name(base.name + "_trace.csv")
        with trace_csv.open("w", encoding="utf-8", newline="") as fh:
            fh.write(data_io.build_trace_table(x, y_raw, fit.get("baseline"), fit.get("peaks_out") or []))
        written.append(trace_csv)
    remove_uncertainty_files(base, keep=keep, log=log)
    if unc_norm is not None:
        written.extend(write_uncertainty(base, file_path, unc_norm, fit, settings))
    return written


# --- the one uncertainty context (game plan 2.1) ---------------------------------


def _parse_int(value) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def uncertainty_workers(settings: Mapping[str, Any]) -> tuple:
    """Uncertainty and band worker counts: 0 ("auto") is one worker, i.e. sequential."""

    cfg_perf = performance.get_parallel_config()
    maxcpu = max(1, (os.cpu_count() or 1))
    req = _parse_int(settings.get("perf_unc_workers", settings.get("unc_workers", settings.get("perf_max_workers", 0))))
    workers = min(maxcpu, max(1, int(cfg_perf.unc_workers))) if req <= 0 else max(1, min(req, maxcpu))
    band_req = _parse_int(settings.get("unc_band_workers", req if req > 0 else cfg_perf.unc_workers))
    band_workers = workers if band_req <= 0 else max(1, min(band_req, maxcpu))
    return workers, band_workers


def bootstrap_band_wanted(settings: Mapping[str, Any]) -> bool:
    export_cfg = dict(settings.get("export", {}) or {})
    unc_section = dict(settings.get("uncertainty", {}) or {})
    need_band = bool(export_cfg.get("include_band", False)) or bool(unc_section.get("return_band", False))
    return bool(settings.get("ui_band_pref_bootstrap", True)) or need_band


def seed_settings(settings: Mapping[str, Any]) -> tuple:
    """(seed all, the seed or None) from the Performance settings."""

    seed_all = bool(settings.get("perf_seed_all", False))
    seed = None
    if seed_all:
        try:
            seed = int(str(settings.get("perf_seed", "")).strip())
        except Exception:
            seed = None
    return seed_all, seed


def bootstrap_draws(settings: Mapping[str, Any]) -> int:
    if "bootstrap_n" not in settings:
        raise KeyError("bootstrap_n missing in batch config")
    try:
        n = int(settings["bootstrap_n"])
    except Exception as exc:
        raise ValueError(f"invalid bootstrap_n: {settings.get('bootstrap_n')!r}") from exc
    if n <= 0:
        raise ValueError(f"bootstrap_n must be > 0 (got {n})")
    return n


def threadpool_limits_for(config: Mapping[str, Any]):
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



def norm_jitter(v, default=0.0):
    """Normalize bootstrap jitter to a fractional value.

    The default is no jitter: every refit starts at the fit. A start jitter
    scales with each parameter's absolute value, so 2 % moves a centre at
    383 cm-1 by 7.7 cm-1 and the refits land in other minima.
    """

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



def compute_uncertainty(
    fit: Mapping[str, Any],
    x,
    y,
    mask,
    settings: Mapping[str, Any],
    method: str,
    *,
    fit_config: Optional[Mapping[str, Any]] = None,
    file_seed: Optional[int] = None,
    abort_event=None,
    log=None,
):
    """One uncertainty method on one fit: the context the GUI and batch share.

    ``fit`` is ``run_fit_consistent``'s result for ``x``, ``y`` over ``mask``
    (with ``return_jacobian`` and ``return_predictors``). ``settings`` holds the
    uncertainty settings: the method's knobs, alpha, seeds, workers and band.
    ``fit_config`` is the configuration the fit ran with, which the bootstrap's
    refits reuse (default: ``settings``), and ``file_seed`` the seed of the fit's
    own restarts, which the refits reuse too. Every method sees what the fit saw:
    the window's points, the fit's baseline, its locks, bounds and noise weights.
    Returns the method's result, raw (see :func:`finish_uncertainty`).
    """

    res = fit
    config = settings
    fit_config = dict(fit_config if fit_config is not None else settings)
    unc_method_canon = data_io.canonical_unc_label(method)
    if unc_method_canon == "unknown":
        raise ValueError(f"Unknown uncertainty method: {method!r}")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    mode = res.get("mode", "add")
    baseline = res.get("baseline")
    baseline = np.zeros_like(x) if baseline is None else np.asarray(baseline, float)
    noise_sigma = res.get("noise_sigma")
    joint_fit = res.get("joint_baseline") is not None
    seed = file_seed
    cfg_perf = performance.get_parallel_config()
    unc_workers, unc_band_workers = uncertainty_workers(settings)
    band_pref_bootstrap = bootstrap_band_wanted(settings)
    perf_seed_all, perf_seed = seed_settings(settings)
    bootstrap_n_cfg = bootstrap_draws(settings) if str(unc_method_canon).lower() == "bootstrap" else None
    _tp_limits_ctx_for_config = threadpool_limits_for
    _norm_jitter = norm_jitter

    theta_hat = np.asarray(res["theta"], float)
    peaks_obj = (
        res.get("peaks_out")
        or res.get("peaks")
        or []
    )
    fit_mode = res.get("mode", mode)
    add_mode = fit_mode == "add"
    # Every method sees what the fit saw: the points in the fit
    # window, in the data space of the fitted model. In add mode
    # that is y with the baseline in the model; in subtract mode it
    # is y minus the baseline, with the peaks alone as the model.
    x_fit = np.asarray(x[mask], float)
    base_window = np.asarray(baseline[mask], float)
    y_fit = np.asarray(y[mask] if add_mode else y[mask] - base_window, float)
    base_fit = base_window if add_mode else np.zeros_like(base_window)

    rj = res.get("residual_jac")
    residual_fn = res.get("residual_fn")
    if residual_fn is None and rj is not None:
        residual_fn = lambda th: rj(th)[0]
    if residual_fn is None:
        residual_fn = build_residual(
            x_fit, y_fit, peaks_obj, "add", base_fit, "linear", None
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

    # One predictor on the fit window serves every method and band:
    # the fit's own, or the same model built here.
    predict_fit_fn = res.get("predict_fit")

    def _predict_fit_local(
        th,
        *,
        _x=x_fit,
        _peaks=peaks_obj,
        _base=base_fit,
    ):
        total = np.zeros_like(_x, float)
        t = np.asarray(th, float).ravel()
        for i in range(len(_peaks)):
            j = 4 * i
            c, h, fw, eta = t[j : j + 4]
            total += models.pseudo_voigt(_x, h, c, fw, eta)
        return total + _base

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
    predict_full_fn = predict_fit_fn

    # The same window model for rows of parameter vectors at once
    # (the Bayesian's vectorised likelihood).
    def predict_batch_fn(rows, *, _x=x_fit, _base=base_fit):
        return models.pseudo_voigt_rows(_x, rows) + _base[None, :]

    bounds_exact = res.get("bounds")
    locked_exact = res.get("locked_mask")
    param_names_unc = res.get("param_names")

    # A joint baseline's coefficients are parameters like the
    # heights: every method gets them after the peaks' (theta,
    # Jacobian columns, predictor, locks, bounds and names), and
    # the Bayesian integrates them out (baseline_basis).
    joint_pb = None
    if joint_fit:
        joint_pb = baseline_model.uncertainty_problem(res, x_fit, y_fit)
        theta_hat = joint_pb["theta"]
        residual_fn = joint_pb["residual_fn"]
        jac = joint_pb["jacobian"]
        predict_fit_fn = predict_full_fn = joint_pb["predict"]
        predict_batch_fn = joint_pb["predict_batch"]
        bounds_exact = joint_pb["bounds"]
        locked_exact = joint_pb["locked_mask"]
        param_names_unc = joint_pb["param_names"]

    fit_ctx = dict(res.get("fit_ctx") or {})
    # Every method takes the sigma_i that weighted the fit.
    fit_ctx["noise_sigma"] = noise_sigma
    if joint_pb is not None:
        fit_ctx["baseline_basis"] = joint_pb["basis"]
        fit_ctx["param_names"] = param_names_unc
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
    if bounds_exact is not None:
        fit_ctx["bounds"] = bounds_exact
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

    jitter_frac = _norm_jitter(config.get("bootstrap_jitter", 0.0))
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
            "predict_batch": predict_batch_fn,
            "x_all": x_fit,
            "y_all": y_fit,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "baseline": base_fit,
            "mode": fit_mode,
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
            # Every method reports at the configured confidence level.
            "alpha": float(config.get("unc_alpha", 0.05)),
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

    # Refits use the file's own solver options and the window's
    # baseline, and start from the start bootstrap_ci hands over.
    refit_cfg = copy.deepcopy(fit_config)
    refit_cfg["solver_choice"] = refit_cfg["solver"] = boot_solver
    # Refits are weighted by the fit's sigma_i, never re-estimated.
    refit_cfg["noise_model"] = "none"
    if noise_sigma is not None:
        refit_cfg["noise_sigma"] = noise_sigma
    if "solver_options" in config:
        _extra = dict(config.get(boot_solver, {}))
        _extra.update(dict(config.get("solver_options") or {}))
        refit_cfg[boot_solver] = _extra

    def _refit_wrapper(theta_init, locked_mask, bounds, x, y):
        """One refit. It counts only if the solver converged; an
        exception is a failed draw, recorded by bootstrap_ci. With a
        joint baseline the refit fits it again over the same window,
        and its coefficients follow the peaks' parameters."""

        t = np.asarray(theta_init, float).ravel()
        peaks_start = copy.deepcopy(peaks_obj)
        for i, pk in enumerate(peaks_start):
            pk.center, pk.height, pk.fwhm, pk.eta = (float(v) for v in t[4 * i : 4 * i + 4])
        # The refit's restarts are seeded like the file's own fit, so a
        # seeded batch gives the same bootstrap on every run.
        out = fit_api.run_fit_consistent(
            x,
            y,
            peaks_start,
            refit_cfg,
            None if joint_pb is not None else base_fit,
            fit_mode,
            np.ones_like(x, bool),
            rng_seed=seed,
        )
        theta_out = np.asarray(out.get("theta"), float).ravel()
        if joint_pb is not None:
            coef = out.get("baseline_coef")
            if coef is None:
                return t, False
            theta_out = np.r_[theta_out, np.asarray(coef, float).ravel()]
        if theta_out.size != t.size:
            return t, False
        return theta_out, bool(out.get("fit_ok")) and bool(np.all(np.isfinite(theta_out)))

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
        # Studentized residuals, r / sqrt(1 - h), by default in batch:
        # measured (PR F, B = 500, 400 paired replicates) it gains 0.6 to 2.2
        # coverage points in every cell and loses none. The core default
        # (GUI) stays off.
        fit_ctx_local["boot_studentize"] = config.get("boot_studentize", True)
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
                    param_names=param_names_unc,
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
        if not band_pref_bootstrap and isinstance(unc_res, dict):
            unc_res["band"] = None  # a band only when one was asked for
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

        # numpy's global state is left alone: the asymptotic draws
        # nothing, and the Bayesian seeds its own generator from seed.
        with _tp_limits_ctx_for_config(config):
            unc_res = route_uncertainty(
                unc_method_canon,
                theta_hat=theta_hat,
                residual_fn=residual_fn,
                jacobian=jac,
                model_eval=predict_fit_fn,
                fit_ctx=fit_ctx,
                x_all=x_fit,
                y_all=y_fit,
                workers=(workers_eff if workers_eff > 0 else None),
                seed=seed_val,
                n_boot=(n_boot if n_boot is not None else 0),
            )
    return unc_res
