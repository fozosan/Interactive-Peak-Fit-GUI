"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import csv
import io
import re
from pathlib import Path

import math
from math import isnan
import numpy as np
import pandas as pd
from collections.abc import Mapping
from types import SimpleNamespace

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

_Z = 1.96  # 95% normal approx


def _to_float_or_nan(x):
    try:
        if x is None:
            return float("nan")
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")


def _ci_from_sd(est, sd, z=_Z):
    est = _to_float_or_nan(est)
    sd = _to_float_or_nan(sd)
    if not math.isfinite(est) or not math.isfinite(sd):
        return (float("nan"), float("nan"))
    return (est - z * sd, est + z * sd)


def _as_mapping(obj):
    """Return an object with .get, handling dict / SimpleNamespace / list fallback."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, SimpleNamespace):
        return obj.__dict__
    if isinstance(obj, list):
        return {"stats": obj}
    return getattr(obj, "__dict__", {}) or {}


def _num(x):
    """Return a clean number or None (for CSV blanks)."""
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _label_from_result(m):
    """Map uncertainty method to a canonical label."""
    method = (m.get("method") or m.get("type") or m.get("label") or "").lower()
    if "asym" in method:
        return "asymptotic"
    if "boot" in method:
        return "bootstrap"
    if "bayes" in method or "mcmc" in method:
        return "bayesian"
    pretty = m.get("label") or m.get("method_label")
    if pretty:
        p = pretty.lower()
        if "asym" in p:
            return "asymptotic"
        if "boot" in p:
            return "bootstrap"
        if "mcmc" in p or "bayes" in p:
            return "bayesian"
    return "unknown"


def _rows_from_stats(file_path, m):
    """
    Convert any uncertainty result into a list of rows for CSV.
    Expected keys per stat: peak, param, est/value, sd/stderr, ci_lo/ci_hi or p2_5/p97_5.
    """
    stats = m.get("stats") or m.get("parameters") or m.get("param_stats") or []
    rows = []
    method_label = _label_from_result(m)
    rmse = _num(m.get("rmse"))
    dof = m.get("dof")

    backend = m.get("backend")
    n_draws = m.get("n_draws") or m.get("n_samples")
    n_boot = m.get("n_boot")
    ess = m.get("ess")
    rhat = m.get("rhat")

    def pick(d, *names):
        for k in names:
            if k in d and d[k] is not None:
                return d[k]
        return None

    for s in stats:
        sdct = s if isinstance(s, dict) else getattr(s, "__dict__", {})
        peak = sdct.get("peak") or sdct.get("index") or sdct.get("k")
        param = sdct.get("param") or sdct.get("name")
        est = pick(sdct, "est", "value", "mean", "median")
        sd = pick(sdct, "sd", "stderr", "std", "stdev")
        p025 = pick(sdct, "p2_5", "q025", "q2_5")
        p975 = pick(sdct, "p97_5", "q975", "q97_5")
        ci_lo = sdct.get("ci_lo")
        ci_hi = sdct.get("ci_hi")
        if ci_lo is None and p025 is not None:
            ci_lo = p025
        if ci_hi is None and p975 is not None:
            ci_hi = p975
        rows.append({
            "file": str(file_path),
            "peak": peak,
            "param": param,
            "value": _num(est),
            "stderr": _num(sd),
            "ci_lo": _num(ci_lo),
            "ci_hi": _num(ci_hi),
            "method": method_label,
            "rmse": rmse,
            "dof": dof,
            "p2_5": _num(p025),
            "p97_5": _num(p975),
            "backend": backend,
            "n_draws": n_draws,
            "n_boot": n_boot,
            "ess": ess,
            "rhat": rhat,
        })
    return rows


def normalize_unc_result(res):
    """
    Normalize any uncertainty container (dict, namespace, list-of-stat-rows)
    to a mapping with:
      method, label, stats(list), rmse, dof, backend, n_draws, n_boot, ess, rhat,
      band (optional 3-tuple), and/or band_x, band_lo, band_hi.
    """
    m = _as_mapping(res).copy()
    if "label" not in m:
        m["label"] = m.get("method_label") or m.get("type") or m.get("method")
    band = m.get("band")
    if band and isinstance(band, (tuple, list)) and len(band) == 3:
        bx, blo, bhi = band
        m["band_x"], m["band_lo"], m["band_hi"] = bx, blo, bhi
    return m


def write_uncertainty_csv(path, res, file_path):
    """
    Write unified CSV:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof, p2_5, p97_5,
      backend, n_draws, n_boot, ess, rhat
    """
    m = normalize_unc_result(res)
    rows = _rows_from_stats(file_path, m)
    cols = ["file","peak","param","value","stderr","ci_lo","ci_hi","method","rmse","dof",
            "p2_5","p97_5","backend","n_draws","n_boot","ess","rhat"]
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") if r.get(k, None) is not None else "" for k in cols})
    return path


def write_uncertainty_txt(path, res, file_path, solver_line, baseline_line, perf_line):
    """
    Human-readable TXT, showing ±stderr and 95% CI (quantiles if available, else ~±1.96σ).
    """
    m = normalize_unc_result(res)
    label = _label_from_result(m)
    rmse = m.get("rmse")
    dof = m.get("dof")
    rows = _rows_from_stats(file_path, m)

    lines = []
    lines.append(f"File: {file_path}")
    lines.append(f"Uncertainty method: {label}")
    if solver_line:   lines.append(f"Solver: {solver_line}")
    if baseline_line: lines.append(f"Baseline: {baseline_line}")
    if perf_line:     lines.append(f"Performance: {perf_line}")
    if rmse is not None and dof is not None:
        lines.append(f"RMSE={rmse:.4g}, dof={dof}")
    lines.append("Peaks:")

    from collections import defaultdict
    g = defaultdict(dict)
    for r in rows:
        g[int(r["peak"]) if r["peak"] is not None else -1][r["param"]] = r

    def _fmt(x, digits=6):
        if x is None: return "n/a"
        try:
            return f"{float(x):.{digits}g}"
        except Exception:
            return "n/a"

    for pk in sorted(g.keys()):
        lines.append(f"Peak {pk}")
        for pname in ("center","height","fwhm","eta"):
            r = g[pk].get(pname)
            if not r:
                continue
            val = _fmt(r.get("value"))
            sd = r.get("stderr")
            ql = r.get("ci_lo")
            qh = r.get("ci_hi")
            if pname == "fwhm" and (sd is None):
                lines.append(f"  {pname:<6} = {val} (fixed)")
            else:
                if ql is None or qh is None:
                    if sd is not None and r.get("value") is not None:
                        try:
                            v = float(r["value"]); s = float(sd)
                            ql = v - 1.96*s
                            qh = v + 1.96*s
                        except Exception:
                            ql = qh = None
                lines.append(f"  {pname:<6} = {val} ± {_fmt(sd,3)}   (95% CI: [{_fmt(ql)}, {_fmt(qh)}])")

    txt = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path


def export_uncertainty_pair(out_csv, out_txt, unc_result, file_path, solver_line, baseline_line, perf_line):
    """
    Write both CSV and TXT for the most recently computed uncertainty result.
    'unc_result' may be dict / namespace / list-of-stats; we normalize inside.
    """
    write_uncertainty_csv(out_csv, unc_result, file_path)
    write_uncertainty_txt(out_txt, unc_result, file_path, solver_line, baseline_line, perf_line)
    return [out_csv, out_txt]


def _pick(d: Dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _stats_for_param(rec: Dict, p: str):
    """Return (est, sd, q_lo, q_hi) from a per-peak record that may be nested or flat."""
    node = rec.get("stats", rec)  # tolerate either
    if isinstance(node.get(p), dict):
        pdct = node[p]
        return (
            _pick(pdct, "est", "mean", default=float("nan")),
            _pick(pdct, "sd", "stderr", "std", default=float("nan")),
            _pick(pdct, "p2_5", "q2_5", "q025", default=float("nan")),
            _pick(pdct, "p97_5", "q97_5", "q975", default=float("nan")),
        )
    return (
        _pick(node, f"{p}_est", default=float("nan")),
        _pick(node, f"{p}_sd", f"{p}_stderr", default=float("nan")),
        _pick(node, f"{p}_p2_5", f"{p}_q2_5", f"{p}_q025", default=float("nan")),
        _pick(node, f"{p}_p97_5", f"{p}_q97_5", f"{p}_q975", default=float("nan")),
    )


def _unc_as_mapping(obj):
    """Map-like view over uncertainty result. List => {'stats': list}."""
    if obj is None:
        return {"stats": None}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"stats": obj}
    d = {}
    for k in (
        "label",
        "method",
        "method_label",
        "type",
        "stats",
        "parameters",
        "param_stats",
        "rmse",
        "dof",
        "backend",
        "n_draws",
        "n_boot",
        "ess",
        "rhat",
        "band",
        "prediction_band",
        "band_x",
        "band_lo",
        "band_hi",
    ):
        if hasattr(obj, k):
            d[k] = getattr(obj, k)
    if not d and hasattr(obj, "__dict__"):
        d = {**obj.__dict__}
    if "stats" not in d and "parameters" in d:
        d["stats"] = d["parameters"]
    return d


def build_uncertainty_rows(
    file_path: str,
    method_label: str,
    rmse: Optional[float],
    dof: Optional[int],
    per_peak_stats: List[Dict],
    method_meta: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build unified long-form rows:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof,
      p2_5, p97_5, backend, n_draws, n_boot, ess, rhat
    method_meta may include backend/n_draws/n_boot/ess/rhat; missing -> empty.
    """
    meta = method_meta or {}
    rows: List[Dict[str, Any]] = []
    mlabel = (method_label or "").strip()

    for k, rec in enumerate(per_peak_stats, 1):
        for param in ("center", "height", "fwhm", "eta"):
            est, sd, qlo, qhi = _stats_for_param(rec, param)

            if math.isfinite(_to_float_or_nan(qlo)) and math.isfinite(_to_float_or_nan(qhi)):
                ci_lo, ci_hi = _to_float_or_nan(qlo), _to_float_or_nan(qhi)
            else:
                ci_lo, ci_hi = _ci_from_sd(est, sd)

            rows.append(
                {
                    "file": str(file_path),
                    "peak": k,
                    "param": param,
                    "value": _to_float_or_nan(est),
                    "stderr": _to_float_or_nan(sd),
                    "ci_lo": _to_float_or_nan(ci_lo),
                    "ci_hi": _to_float_or_nan(ci_hi),
                    "method": mlabel.lower().replace(" (jᵀj)", "").replace(" (j^tj)", ""),
                    "rmse": _to_float_or_nan(rmse),
                    "dof": _to_float_or_nan(dof),
                    "p2_5": _to_float_or_nan(qlo),
                    "p97_5": _to_float_or_nan(qhi),
                    "backend": meta.get("backend", ""),
                    "n_draws": meta.get("n_draws", ""),
                    "n_boot": meta.get("n_boot", ""),
                    "ess": meta.get("ess", ""),
                    "rhat": meta.get("rhat", ""),
                }
            )
    return rows


def _write_uncertainty_csv(path: str, rows: List[Dict[str, Any]]):
    df = pd.DataFrame(
        rows,
        columns=[
            "file",
            "peak",
            "param",
            "value",
            "stderr",
            "ci_lo",
            "ci_hi",
            "method",
            "rmse",
            "dof",
            "p2_5",
            "p97_5",
            "backend",
            "n_draws",
            "n_boot",
            "ess",
            "rhat",
        ],
    )
    df.to_csv(path, index=False, lineterminator="\n")


def _write_uncertainty_txt(
    path: str,
    file_path: str,
    method_label: str,
    solver_meta: str,
    baseline_meta: str,
    perf_meta: str,
    per_peak_stats: List[Dict],
    z: float = _Z,
):
    def fmt(v, nd=6):
        f = _to_float_or_nan(v)
        return "n/a" if not math.isfinite(f) else f"{f:.{nd}g}"

    lines = []
    lines.append(f"File: {file_path}")
    lines.append(f"Uncertainty method: {method_label.lower()}")
    lines.append(solver_meta)
    lines.append(baseline_meta)
    lines.append(perf_meta)
    lines.append("Peaks:")
    for k, rec in enumerate(per_peak_stats, 1):
        lines.append(f"Peak {k}")
        for param, label in (("center", "center"), ("height", "height"), ("fwhm", "fwhm"), ("eta", "eta")):
            est, sd, qlo, qhi = _stats_for_param(rec, param)
            if math.isfinite(_to_float_or_nan(qlo)) and math.isfinite(_to_float_or_nan(qhi)):
                lo, hi = qlo, qhi
            else:
                lo, hi = _ci_from_sd(est, sd, z)
            if param == "fwhm" and rec.get("lock_width") is True:
                lines.append(f"  {label:<6}= {fmt(est)} (fixed)")
            elif param == "center" and rec.get("lock_center") is True:
                lines.append(f"  {label:<6}= {fmt(est)} (fixed)")
            else:
                lines.append(f"  {label:<6}= {fmt(est)} \u00b1 {fmt(sd)}   (95% CI: [{fmt(lo)}, {fmt(hi)}])")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
def _rows_to_per_peak_stats(result, peaks):
    by_peak: Dict[int, Dict[str, Any]] = {}
    for r in _iter_peak_param_stats(result, peaks or []):
        pk = by_peak.setdefault(r["peak"], {})
        pk.setdefault("stats", {})
        pk["stats"].setdefault(r["param"], {})
        if r.get("value") is not None:
            pk["stats"][r["param"]]["est"] = r.get("value")
            pk[r["param"]] = r.get("value")
        if r.get("stderr") is not None:
            pk["stats"][r["param"]]["sd"] = r.get("stderr")
        if r.get("p2_5") is not None:
            pk["stats"][r["param"]]["p2_5"] = r.get("p2_5")
        if r.get("p97_5") is not None:
            pk["stats"][r["param"]]["p97_5"] = r.get("p97_5")
        if r["param"] == "center":
            pk["lock_center"] = r.get("locked", False)
        if r["param"] == "fwhm":
            pk["lock_width"] = r.get("locked", False)
    return [by_peak[k] for k in sorted(by_peak.keys())]


def _iter_param_rows(unc_res, peaks, method_label: str):
    """Yield normalized per-parameter rows for legacy interfaces."""
    per_peak = _rows_to_per_peak_stats(unc_res, peaks)
    for idx, rec in enumerate(per_peak, 1):
        for param in ("center", "height", "fwhm", "eta"):
            est, sd, qlo, qhi = _stats_for_param(rec, param)
            yield {
                "peak": idx,
                "param": param,
                "est": est,
                "sd": sd,
                "p2_5": qlo,
                "p97_5": qhi,
                "method": method_label,
            }

def _unc_normalize(obj):
    """
    Return a dict-like uncertainty payload with at least:
      - 'label' (str)         : human-readable method label
      - 'method' (str)        : short method key, e.g. 'asymptotic','bootstrap','bayesian'
      - 'stats' (list[dict])  : each dict has: peak,param, value|est, stderr|sd, ci_lo, ci_hi, p2_5, p97_5
      - 'diagnostics' (dict)  : optional metadata (backend, n_draws, n_boot, ess, rhat, rmse, dof, etc)
      - 'band' (tuple|None)   : (x, lo, hi) or None
    Accepts Mapping, SimpleNamespace, or list-of-rows. Never returns a list.
    """
    # Already a mapping
    if isinstance(obj, Mapping):
        return dict(obj)

    # Namespace -> dict
    if isinstance(obj, SimpleNamespace):
        return vars(obj)

    # List/tuple of row dicts -> wrap
    if isinstance(obj, (list, tuple)):
        # assume it's a stats row list
        return {'label': 'unknown', 'method': 'unknown', 'stats': list(obj), 'diagnostics': {}, 'band': None}

    # Fallback to empty structure
    return {'label': 'unknown', 'method': 'unknown', 'stats': [], 'diagnostics': {}, 'band': None}


def _row_value(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _build_unc_rows(unc_mapping, file_path, rmse=None, dof=None):
    """
    Build normalized CSV rows from an uncertainty mapping.
    Returns list of dict rows with fixed schema:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof,
      p2_5, p97_5, backend, n_draws, n_boot, ess, rhat
    """
    u = _unc_normalize(unc_mapping)
    label   = u.get('label') or u.get('method') or 'unknown'
    method  = (u.get('method') or label or 'unknown').lower()
    stats   = u.get('stats') or []
    diag    = u.get('diagnostics') or {}
    file_path = "" if file_path is None else file_path

    rows = []
    for r in stats:
        # Support both flat and nested per-param dicts
        if isinstance(r, Mapping) and 'param' in r:
            peak   = int(_row_value(r, 'peak', default=0) or 0)
            param  = str(_row_value(r, 'param', default=''))
            value  = _row_value(r, 'value', 'est')
            stderr = _row_value(r, 'stderr', 'sd')
            ci_lo  = _row_value(r, 'ci_lo', 'p2_5')
            ci_hi  = _row_value(r, 'ci_hi', 'p97_5')
            p2_5   = _row_value(r, 'p2_5')
            p97_5  = _row_value(r, 'p97_5')
        else:
            # unknown row type -> skip
            continue

        rows.append({
            'file': file_path,
            'peak': peak,
            'param': param,
            'value': value,
            'stderr': stderr,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'method': method,
            'rmse': rmse,
            'dof': dof,
            'p2_5': p2_5,
            'p97_5': p97_5,
            'backend': diag.get('backend'),
            'n_draws': diag.get('n_draws') or diag.get('n_samples'),
            'n_boot': diag.get('n_boot') or diag.get('n_resamples'),
            'ess': diag.get('ess'),
            'rhat': diag.get('rhat'),
        })
    return rows


def export_uncertainty_csv(path, unc_result, file_path, rmse=None, dof=None):
    """
    Write uncertainty rows to CSV. Accepts Mapping/Namespace/list; never assumes .get().
    """
    rows = _build_unc_rows(unc_result, file_path, rmse=rmse, dof=dof)
    import csv
    fieldnames = ['file','peak','param','value','stderr','ci_lo','ci_hi','method','rmse','dof',
                  'p2_5','p97_5','backend','n_draws','n_boot','ess','rhat']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def export_uncertainty_txt(path, unc_result, meta=None):
    """
    Human-readable TXT export. Accepts any result shape.
    meta may contain: solver, baseline, performance, rmse, dof, file.
    """
    u = _unc_normalize(unc_result)
    label  = u.get('label') or u.get('method') or 'unknown'
    stats  = u.get('stats') or []
    diag   = u.get('diagnostics') or {}
    rmse   = (meta or {}).get('rmse')
    dof    = (meta or {}).get('dof')
    file_  = (meta or {}).get('file')

    def fmt(x, nan='n/a'):
        try:
            if x is None:
                return nan
            return f"{float(x):.6g}"
        except Exception:
            return str(x) if x is not None else nan

    lines = []
    if file_: lines.append(f"File: {file_}")
    lines.append(f"Uncertainty method: {label}")
    if meta and meta.get('solver'):
        lines.append(meta['solver'])
    if meta and meta.get('baseline'):
        lines.append(meta['baseline'])
    if meta and meta.get('performance'):
        lines.append(meta['performance'])
    if rmse is not None or dof is not None:
        lines.append(f"RMSE={fmt(rmse)}, dof={fmt(dof)}")
    if diag:
        # optional method diagnostics if present
        parts = []
        for k in ('backend','n_draws','n_boot','ess','rhat'):
            if k in diag and diag[k] is not None:
                parts.append(f"{k}={diag[k]}")
        if parts:
            lines.append("Diagnostics: " + ", ".join(parts))

    lines.append("Peaks:")
    # Group by peak
    from collections import defaultdict
    by_peak = defaultdict(list)
    for r in stats:
        if isinstance(r, Mapping) and 'param' in r:
            by_peak[int(r.get('peak', 0))].append(r)

    for k in sorted(by_peak.keys()):
        lines.append(f"Peak {k}")
        rows = by_peak[k]
        # write known params in nice order
        order = ['center','height','fwhm','eta']
        for p in order:
            rr = next((r for r in rows if r.get('param') == p), None)
            if rr is None: 
                continue
            val = _row_value(rr, 'value', 'est')
            sd  = _row_value(rr, 'stderr', 'sd')
            lo  = _row_value(rr, 'ci_lo', 'p2_5')
            hi  = _row_value(rr, 'ci_hi', 'p97_5')
            fixed = rr.get('fixed', False) or rr.get('lock', False)
            if fixed:
                lines.append(f"  {p:<6} = {fmt(val)} (fixed)")
            else:
                lines.append(f"  {p:<6} = {fmt(val)} ± {fmt(sd)}   (95% CI: [{fmt(lo)}, {fmt(hi)}])")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return path


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
    stats = list(_iter_param_rows(unc_res, peaks, method_label or ""))
    mapping = {
        'label': method_label or 'unknown',
        'method': method_label or 'unknown',
        'stats': stats,
        'diagnostics': {},
    }
    return export_uncertainty_csv(path, mapping, file_path, rmse=rmse, dof=dof)


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
    stats = list(_iter_param_rows(unc_res, peaks, method_label or ""))
    mapping = {
        'label': method_label or 'unknown',
        'method': method_label or 'unknown',
        'stats': stats,
        'diagnostics': {},
    }
    meta = {
        'file': file_path,
        'solver': solver_meta,
        'baseline': baseline_meta,
        'performance': perf_meta,
    }
    return export_uncertainty_txt(path, mapping, meta=meta)


