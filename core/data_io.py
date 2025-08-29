"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union, Any, Optional, List

import csv
import io
import re
from pathlib import Path

import math
import numpy as np
import pandas as pd

from .uncertainty import UncertaintyResult


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load two-column numeric data from ``path``.

    The function accepts ``.txt``, ``.csv`` or ``.dat`` files containing two
    numeric columns. Delimiters (comma, tab, semicolon or whitespace) are
    autodetected and lines beginning with common comment prefixes (``#``, ``%``,
    ``//``) or header strings are ignored.
    """

    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # strip out comments (full-line or inline)
            for cc in ("#", "%", "//"):
                if line.startswith(cc):
                    line = ""
                    break
                if cc in line:
                    line = line.split(cc, 1)[0].strip()
            if not line:
                continue
            parts = re.split(r"[,\s;]+", line)
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
            except (IndexError, ValueError):
                continue
            xs.append(x_val)
            ys.append(y_val)

    if len(xs) < 2:
        raise ValueError("Could not parse a two-column numeric dataset from the file.")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size >= 2 and x[1] < x[0]:
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
    return x, y


def derive_export_paths(user_path: str) -> dict:
    """Return canonical export file paths based on ``user_path``.

    ``user_path`` may have any extension; the returned paths drop the
    extension and append ``_fit.csv``, ``_trace.csv`` and the uncertainty
    artefacts.
    """

    p = Path(user_path)
    base = p.with_suffix("")
    return {
        "fit": base.with_name(base.name + "_fit.csv"),
        "trace": base.with_name(base.name + "_trace.csv"),
        "unc_txt": base.with_name(base.name + "_uncertainty.txt"),
        "unc_csv": base.with_name(base.name + "_uncertainty.csv"),
        "unc_band": base.with_name(base.name + "_uncertainty_band.csv"),
    }


def build_peak_table(records: Iterable[dict]) -> str:
    """Return a CSV-formatted peak table built from ``records``.

    ``records`` should provide the columns defined in the blueprint. Extra keys
    are ignored so that future schema extensions remain compatible.
    """

    headers = [
        "file",
        "peak",
        "center",
        "height",
        "fwhm",
        "eta",
        "lock_width",
        "lock_center",
        "area",
        "area_pct",
        "rmse",
        "fit_ok",
        "mode",
        "als_lam",
        "als_p",
        "fit_xmin",
        "fit_xmax",
        "solver_choice",
        "solver_loss",
        "solver_weight",
        "solver_fscale",
        "solver_maxfev",
        "solver_restarts",
        "solver_jitter_pct",
        "use_baseline",
        "baseline_mode",
        "baseline_uses_fit_range",
        "als_niter",
        "als_thresh",
        "perf_numba",
        "perf_gpu",
        "perf_cache_baseline",
        "perf_seed_all",
        "perf_max_workers",
        "bounds_center_lo",
        "bounds_center_hi",
        "bounds_fwhm_lo",
        "bounds_height_lo",
        "bounds_height_hi",
        "x_scale",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=headers, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    return buf.getvalue()


def build_trace_table(
    x: np.ndarray,
    y_raw: np.ndarray,
    baseline: np.ndarray | None,
    peaks: Iterable,
) -> str:
    """Return a CSV trace table matching the v2.7 schema.

    Columns (in order): ``x, y_raw, baseline, y_target_add, y_fit_add,``
    per-peak additive components, ``y_target_sub, y_fit_sub`` and per-peak
    subtractive components. The builder always emits both additive and
    subtractive sections even when no peaks are present.
    """

    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    base = np.asarray(baseline, dtype=float) if baseline is not None else np.zeros_like(x)

    from .models import pv_sum  # local import to avoid cycles

    comps = [pv_sum(x, [p]) for p in peaks]
    comps_arr = np.vstack(comps) if comps else np.empty((0, x.size))
    model = comps_arr.sum(axis=0) if comps else np.zeros_like(x)

    y_target_add = y_raw
    y_fit_add = model + base
    y_target_sub = y_raw - base
    y_fit_sub = model

    headers = ["x", "y_raw", "baseline", "y_target_add", "y_fit_add"]
    headers += [f"peak{i+1}" for i in range(comps_arr.shape[0])]
    headers += ["y_target_sub", "y_fit_sub"]
    headers += [f"peak{i+1}_sub" for i in range(comps_arr.shape[0])]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for idx in range(x.size):
        row = [
            x[idx],
            y_raw[idx],
            base[idx],
            y_target_add[idx],
            y_fit_add[idx],
        ]
        row.extend((base[idx] + comps_arr[:, idx]).tolist() if comps else [])
        row.append(y_target_sub[idx])
        row.append(y_fit_sub[idx])
        row.extend(comps_arr[:, idx].tolist() if comps else [])
        writer.writerow(row)
    return buf.getvalue()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` without introducing extra blank lines."""

    with path.open("w", newline="", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


_Z = 1.96  # 95% normal


def _normalize_band(result: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return (x, lo, hi) arrays or None.
    Accepts UncertaintyResult (.band/.prediction_band) or dict {'band'|'prediction_band'|'ci_band': (x, lo, hi)}.
    """
    band = None
    if result is None:
        return None
    band = getattr(result, "band", None) or getattr(result, "prediction_band", None)
    if band is None and isinstance(result, dict):
        band = result.get("band") or result.get("prediction_band") or result.get("ci_band")
    if band is None:
        return None
    try:
        if len(band) < 3:
            return None
        x, lo, hi = band[0], band[1], band[2]
        x = np.asarray(x); lo = np.asarray(lo); hi = np.asarray(hi)
        if x.shape != lo.shape or x.shape != hi.shape or x.size == 0:
            return None
        return x, lo, hi
    except Exception:
        return None


def _method_label(res: Any, default: str = "Unknown") -> str:
    for key in ("label", "method_label", "method", "type"):
        v = getattr(res, key, None) if not isinstance(res, dict) else res.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return default


def _pack_stats_for_param(param: str, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize per-param stats from result.stats or similar:
      expect keys like: est/value, sd/stderr, ci_lo, ci_hi, p2_5, p97_5
    """
    # prefer common aliases
    est = stats_dict.get("est", stats_dict.get("value"))
    sd = stats_dict.get("sd", stats_dict.get("stderr"))
    p2 = stats_dict.get("p2_5")
    p97 = stats_dict.get("p97_5")
    ci_lo = stats_dict.get("ci_lo")
    ci_hi = stats_dict.get("ci_hi")
    # if missing CI, try normal approx
    if ci_lo is None and ci_hi is None and est is not None and sd is not None:
        try:
            ci_lo = float(est) - _Z * float(sd)
            ci_hi = float(est) + _Z * float(sd)
        except Exception:
            ci_lo = None; ci_hi = None
    return dict(param=param, value=est, stderr=sd, ci_lo=ci_lo, ci_hi=ci_hi, p2_5=p2, p97_5=p97)


def _iter_peak_param_stats(result: Any, peaks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build normalized param rows for each peak, covering center, height, fwhm, eta.
    Handles locked/fixed parameters by emitting stderr/ci as None and marking value as current.
    """
    # Find stats container:
    stats_container = None
    if result is not None:
        stats_container = getattr(result, "stats", None)
        if stats_container is None and isinstance(result, dict):
            stats_container = result.get("stats") or result.get("parameters") or result.get("param_stats")
    rows = []
    # We expect 'peaks' entries to have current values + lock flags
    for i, pk in enumerate(peaks, 1):
        # Look up stats per param if present; else populate with current values and None for sd/ci
        for param in ("center", "height", "fwhm", "eta"):
            if isinstance(pk, dict):
                current = pk.get(param)
                lock_key = f"lock_{'center' if param=='center' else 'width' if param=='fwhm' else 'none'}"
                locked = bool(pk.get(lock_key, False)) if lock_key != "lock_none" else False
            else:
                current = getattr(pk, param, None)
                lock_attr = 'lock_center' if param == 'center' else 'lock_width' if param == 'fwhm' else None
                locked = bool(getattr(pk, lock_attr, False)) if lock_attr else False
            sd = None; ci_lo = None; ci_hi = None; p2 = None; p97 = None
            if stats_container:
                # stats may be structure: stats[i-1][param] -> dict
                per_peak = None
                if isinstance(stats_container, list):
                    per_peak = stats_container[i-1] if i-1 < len(stats_container) else None
                elif isinstance(stats_container, dict):
                    per_peak = stats_container.get(i) or stats_container.get(str(i))
                if per_peak and isinstance(per_peak, dict):
                    stat_block = per_peak.get(param)
                    if isinstance(stat_block, dict):
                        packed = _pack_stats_for_param(param, stat_block)
                        current = packed["value"] if packed["value"] is not None else current
                        sd = packed["stderr"]; ci_lo = packed["ci_lo"]; ci_hi = packed["ci_hi"]
                        p2 = packed["p2_5"]; p97 = packed["p97_5"]
                    else:
                        # flat form: center_est/center_sd...
                        est = per_peak.get(f"{param}_est")
                        sd = per_peak.get(f"{param}_sd", per_peak.get(f"{param}_stderr"))
                        p2 = per_peak.get(f"{param}_p2_5")
                        p97 = per_peak.get(f"{param}_p97_5")
                        ci_lo = per_peak.get(f"{param}_ci_lo")
                        ci_hi = per_peak.get(f"{param}_ci_hi")
                        if est is not None:
                            current = est
                        if ci_lo is None and ci_hi is None and est is not None and sd is not None:
                            try:
                                ci_lo = float(est) - _Z * float(sd)
                                ci_hi = float(est) + _Z * float(sd)
                            except Exception:
                                pass
            rows.append(dict(peak=i, param=param, value=current, stderr=sd,
                             ci_lo=ci_lo, ci_hi=ci_hi, p2_5=p2, p97_5=p97, locked=locked))
    return rows


class _DictResult(UncertaintyResult):
    """Shim exposing custom method labels for legacy dict results."""

    def __init__(
        self,
        method: str,
        band,
        param_stats: Dict[str, Dict[str, float]],
        meta: Dict[str, object],
        label: str,
    ) -> None:
        super().__init__(method, band, param_stats, meta)
        self._label = label

    @property
    def method_label(self) -> str:  # type: ignore[override]
        return self._label


def _ensure_result(unc: Union[UncertaintyResult, dict]) -> UncertaintyResult:
    if isinstance(unc, UncertaintyResult):
        return unc

    method = str(unc.get("method") or unc.get("type") or "unknown").lower()
    if method == "asymptotic":
        method_label = "Asymptotic (JᵀJ)"
    elif method == "bootstrap":
        method_label = "Bootstrap (residual)"
    elif method == "bayesian":
        method_label = "Bayesian (MCMC)"
    else:
        method_label = "unknown"

    params: Dict[str, Dict[str, float]] = {}
    for name, stats in unc.get("params", {}).items():
        est = stats.get("est")
        if est is None:
            est = stats.get("mean")
        if est is None:
            est = stats.get("median")
        sd = stats.get("sd")
        if sd is None:
            sd = stats.get("stderr")
        if sd is None:
            sd = stats.get("sigma")
        p2 = stats.get("p2.5")
        if p2 is None:
            p2 = stats.get("p2_5")
        if p2 is None:
            p2 = stats.get("q05")
        p97 = stats.get("p97.5")
        if p97 is None:
            p97 = stats.get("p97_5")
        if p97 is None:
            p97 = stats.get("q95")
        params[name] = {"est": est, "sd": sd}
        if p2 is not None and p97 is not None:
            params[name]["p2.5"] = p2
            params[name]["p97.5"] = p97

    band = _normalize_band(unc)

    diagnostics = {
        "ess": unc.get("diagnostics", {}).get("ess"),
        "rhat": unc.get("diagnostics", {}).get("rhat"),
    }
    return UncertaintyResult(
        method=method,
        label=method_label,
        stats=params,
        diagnostics=diagnostics,
        band=band,
    )

def _iter_param_rows(unc_res, peaks, method_label: str):
    """Yield normalized per-parameter rows for uncertainty exports."""

    stats = getattr(unc_res, "stats", None)
    if stats is None and isinstance(unc_res, dict):
        stats = unc_res.get("stats")
        if stats is None:
            stats = unc_res.get("parameters")
        if stats is None:
            stats = unc_res.get("param_stats")
    if not stats:
        return

    # support both per-peak and per-parameter layouts
    if any(isinstance(v, dict) and isinstance(v.get("est"), (list, tuple, np.ndarray)) for v in stats.values()):
        centers = stats.get("center", {})
        heights = stats.get("height", {})
        fwhms = stats.get("fwhm", {})
        etas = stats.get("eta", {})

        est_c = centers.get("est")
        if est_c is None:
            est_c = []
        est_h = heights.get("est")
        if est_h is None:
            est_h = []
        est_w = fwhms.get("est")
        if est_w is None:
            est_w = []

        sd_c = centers.get("sd")
        if sd_c is None:
            sd_c = []
        sd_h = heights.get("sd")
        if sd_h is None:
            sd_h = []
        sd_w = fwhms.get("sd")
        if sd_w is None:
            sd_w = []
        est_e = etas.get("est")
        if est_e is None:
            est_e = []
        sd_e = etas.get("sd")
        if sd_e is None:
            sd_e = []

        p2_c = centers.get("p2_5")
        if p2_c is None:
            p2_c = centers.get("p2.5")
        if p2_c is None:
            p2_c = []
        p2_h = heights.get("p2_5")
        if p2_h is None:
            p2_h = heights.get("p2.5")
        if p2_h is None:
            p2_h = []
        p2_w = fwhms.get("p2_5")
        if p2_w is None:
            p2_w = fwhms.get("p2.5")
        if p2_w is None:
            p2_w = []
        p2_e = etas.get("p2_5")
        if p2_e is None:
            p2_e = etas.get("p2.5")
        if p2_e is None:
            p2_e = []

        p97_c = centers.get("p97_5")
        if p97_c is None:
            p97_c = centers.get("p97.5")
        if p97_c is None:
            p97_c = []
        p97_h = heights.get("p97_5")
        if p97_h is None:
            p97_h = heights.get("p97.5")
        if p97_h is None:
            p97_h = []
        p97_w = fwhms.get("p97_5")
        if p97_w is None:
            p97_w = fwhms.get("p97.5")
        if p97_w is None:
            p97_w = []
        p97_e = etas.get("p97_5")
        if p97_e is None:
            p97_e = etas.get("p97.5")
        if p97_e is None:
            p97_e = []
        for i, _ in enumerate(peaks, 1):
            yield {
                "peak": i,
                "param": "center",
                "est": _safe_idx(est_c, i - 1),
                "sd": _safe_idx(sd_c, i - 1),
                "p2_5": _safe_idx(p2_c, i - 1),
                "p97_5": _safe_idx(p97_c, i - 1),
                "method": method_label,
            }
            yield {
                "peak": i,
                "param": "height",
                "est": _safe_idx(est_h, i - 1),
                "sd": _safe_idx(sd_h, i - 1),
                "p2_5": _safe_idx(p2_h, i - 1),
                "p97_5": _safe_idx(p97_h, i - 1),
                "method": method_label,
            }
            yield {
                "peak": i,
                "param": "fwhm",
                "est": _safe_idx(est_w, i - 1),
                "sd": _safe_idx(sd_w, i - 1),
                "p2_5": _safe_idx(p2_w, i - 1),
                "p97_5": _safe_idx(p97_w, i - 1),
                "method": method_label,
            }
            yield {
                "peak": i,
                "param": "eta",
                "est": _safe_idx(est_e, i - 1),
                "sd": _safe_idx(sd_e, i - 1),
                "p2_5": _safe_idx(p2_e, i - 1),
                "p97_5": _safe_idx(p97_e, i - 1),
                "method": method_label,
            }
    else:
        for i, _ in enumerate(peaks, 1):
            s = stats.get(i) or stats.get(str(i)) or {}
            for pname in ("center", "height", "fwhm", "eta"):
                pdict = s.get(pname) or {}
                yield {
                    "peak": i,
                    "param": pname,
                    "est": pdict.get("est"),
                    "sd": pdict.get("sd"),
                    "p2_5": pdict.get("p2_5") or pdict.get("p2.5"),
                    "p97_5": pdict.get("p97_5") or pdict.get("p97.5"),
                    "method": method_label,
                }


def _safe_idx(arr, idx):
    try:
        return arr[idx]
    except Exception:
        return None


def export_uncertainty_csv(
    out_path: str | Path,
    file_path: str | Path | None = None,
    method_label: str = "",
    rmse: float | None = None,
    dof: float | None = None,
    peaks: Iterable[Dict[str, Any]] | None = None,
    result: Any = None,
) -> str | Path:
    """
    Writes a long-form CSV with legacy columns:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof
    and (if present) optional columns: p2_5, p97_5, backend, n_draws, n_boot, ess, rhat.
    """
    import csv

    # Backward compatibility: old signature (path, result, peaks=None, method_label="")
    if result is None and peaks is None and not isinstance(file_path, (str, Path)):
        result = file_path
        file_path = None

    rows = []
    norm_rows = _iter_peak_param_stats(result, peaks or [])  # uses current values if stats missing
    # Optional diagnostics:
    diag = getattr(result, "diagnostics", None) if result is not None and not isinstance(result, dict) else (result.get("diagnostics") if isinstance(result, dict) else None)
    backend = None; n_draws = None; n_boot = None; ess = None; rhat = None
    if isinstance(diag, dict):
        backend = diag.get("backend")
        n_draws = diag.get("n_draws")
        n_boot = diag.get("n_boot")
        ess = diag.get("ess")
        rhat = diag.get("rhat")

    for r in norm_rows:
        rows.append({
            "file": str(file_path) if file_path else "",
            "peak": r["peak"],
            "param": r["param"],
            "value": r["value"],
            "stderr": r["stderr"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "method": method_label,
            "rmse": rmse,
            "dof": dof,
            # optional extras
            "p2_5": r.get("p2_5"),
            "p97_5": r.get("p97_5"),
            "backend": backend,
            "n_draws": n_draws,
            "n_boot": n_boot,
            "ess": ess,
            "rhat": rhat,
        })

    # ensure consistent column order
    fieldnames = ["file","peak","param","value","stderr","ci_lo","ci_hi","method","rmse","dof",
                  "p2_5","p97_5","backend","n_draws","n_boot","ess","rhat"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    return out_path


def export_uncertainty_txt(
    out_path: str | Path,
    file_path: str | Path | None = None,
    method_label: str = "",
    solver_meta: Dict[str, Any] | None = None,
    baseline_meta: Dict[str, Any] | None = None,
    perf_meta: Dict[str, Any] | None = None,
    peaks: Iterable[Dict[str, Any]] | None = None,
    result: Any = None,
    z: float = 1.96,
) -> str | Path:
    """
    Writes a human-readable report mirroring the legacy format.
    Expects:
      solver_meta: dict with solver, loss, weight, f_scale, maxfev, restarts, jitter_pct
      baseline_meta: dict with uses_fit_range, lam, p, niter, thresh
      perf_meta: dict with numba, gpu, cache_baseline, seed_all, max_workers
    """
    lines = []
    lines.append(f"File: {file_path}")
    # Method line
    if "Asymptotic" in method_label:
        lines.append(f"Uncertainty method: {method_label} (95% CI, z={z})")
    else:
        lines.append(f"Uncertainty method: {method_label}")
    # Solver/baseline/perf
    s = solver_meta or {}
    lines.append("Solver: {solver}{loss}{weight}{f}{mfev}{rs}{jit}".format(
        solver=s.get("solver","unknown"),
        loss=f", loss={s.get('loss')}" if s.get('loss') is not None else "",
        weight=f", weight={s.get('weight')}" if s.get('weight') is not None else "",
        f=f", f_scale={s.get('f_scale')}" if s.get('f_scale') is not None else "",
        mfev=f", maxfev={s.get('maxfev')}" if s.get('maxfev') is not None else "",
        rs=f", restarts={s.get('restarts')}" if s.get('restarts') is not None else "",
        jit=f", jitter_pct={s.get('jitter_pct')}" if s.get('jitter_pct') is not None else "",
    ))
    b = baseline_meta or {}
    lines.append("Baseline: uses_fit_range={uses} , lam={lam} , p={p} , niter={niter} , thresh={th}".format(
        uses=b.get("uses_fit_range", False),
        lam=b.get("lam"),
        p=b.get("p"),
        niter=b.get("niter"),
        th=b.get("thresh"),
    ))
    pmeta = perf_meta or {}
    lines.append("Performance: numba={numba}, gpu={gpu}, cache_baseline={cache}, seed_all={seed}, max_workers={mw}".format(
        numba=pmeta.get("numba"), gpu=pmeta.get("gpu"),
        cache=pmeta.get("cache_baseline"), seed=pmeta.get("seed_all"),
        mw=pmeta.get("max_workers"),
    ))
    lines.append("Peaks:")

    # Build normalized stats for printing
    norm_rows = _iter_peak_param_stats(result, peaks or [])
    # group by peak
    by_peak: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in norm_rows:
        by_peak.setdefault(r["peak"], {})[r["param"]] = r

    def _fmt_val_sd_ci(v, sd, lo, hi):
        def _fmt(x, n=6):
            try:
                return f"{float(x):.6g}"
            except Exception:
                return "n/a"
        # if sd None and lo/hi provided, keep ± as missing
        if v is None and sd is None and lo is None and hi is None:
            return "n/a"
        v_s = _fmt(v)
        sd_s = _fmt(sd) if sd is not None else "n/a"
        lo_s = _fmt(lo) if lo is not None else "n/a"
        hi_s = _fmt(hi) if hi is not None else "n/a"
        return f"{v_s} ± {sd_s}   (95% CI: [{lo_s}, {hi_s}])"

    for k in sorted(by_peak.keys()):
        lines.append(f"Peak {k}")
        row_c = by_peak[k].get("center",  {})
        row_h = by_peak[k].get("height", {})
        row_w = by_peak[k].get("fwhm",   {})
        row_e = by_peak[k].get("eta",    {})
        # Handle fixed width/center display
        locked_w = row_w.get("locked", False)
        locked_c = row_c.get("locked", False)
        if locked_c:
            center_line = f"  center = {row_c.get('value','n/a')} (fixed)"
        else:
            center_line = "  center = " + _fmt_val_sd_ci(row_c.get("value"), row_c.get("stderr"),
                                                         row_c.get("ci_lo"), row_c.get("ci_hi"))
        if locked_w:
            width_line  = f"  fwhm   = {row_w.get('value','n/a')} (fixed)"
        else:
            width_line  = "  fwhm   = " + _fmt_val_sd_ci(row_w.get("value"), row_w.get("stderr"),
                                                         row_w.get("ci_lo"), row_w.get("ci_hi"))
        height_line = "  height = " + _fmt_val_sd_ci(row_h.get("value"), row_h.get("stderr"),
                                                     row_h.get("ci_lo"), row_h.get("ci_hi"))
        eta_line    = "  eta    = " + _fmt_val_sd_ci(row_e.get("value"), row_e.get("stderr"),
                                                     row_e.get("ci_lo"), row_e.get("ci_hi"))
        lines.extend([center_line, height_line, width_line, eta_line])

    txt = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return out_path


# Backwards compatible aliases with older API names
def write_uncertainty_csv(path, unc_res, peaks=None, method_label: str = "", rmse=None, dof=None, file_path=None):
    if peaks is None:
        res = _ensure_result(unc_res)
        row: Dict[str, float | str] = {"method": res.method_label}
        for name, stats in res.param_stats.items():
            row[f"{name}_est"] = stats.get("est")
            row[f"{name}_sd"] = stats.get("sd")
            if "p2.5" in stats and "p97.5" in stats:
                row[f"{name}_p2_5"] = stats.get("p2.5")
                row[f"{name}_p97_5"] = stats.get("p97.5")
        df = pd.DataFrame([row])
        write_dataframe(df, Path(path))
        return path
    return export_uncertainty_csv(path, file_path, method_label, rmse, dof, peaks, unc_res)


def write_uncertainty_txt(path, unc_res, peaks=None, method_label: str = "", file_path=None, solver_meta=None, baseline_meta=None, perf_meta=None):
    if peaks is None:
        res = _ensure_result(unc_res)
        lines = [f"Method: {res.method_label}"]
        for name, stats in res.param_stats.items():
            est = stats.get("est")
            sd = stats.get("sd")
            line = f"{name}: {est:.6g} ± {sd:.6g}" if est is not None and sd is not None else f"{name}: n/a"
            if "p2.5" in stats and "p97.5" in stats:
                line += f"   [2.5%: {stats['p2.5']:.6g}, 97.5%: {stats['p97.5']:.6g}]"
            lines.append(line)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    return export_uncertainty_txt(path, file_path, method_label, solver_meta or {}, baseline_meta or {}, perf_meta or {}, peaks, unc_res)


