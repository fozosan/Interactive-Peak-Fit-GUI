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
        x = np.asarray(x)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
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


def export_uncertainty_csv(
    out_path: str | Path,
    file_path: str | Path | None = None,
    method_label: str = "",
    rmse: float | None = None,
    dof: float | None = None,
    peaks: Iterable[Dict[str, Any]] | None = None,
    result: Any = None,
) -> str | Path:
    """Compatibility wrapper producing a unified uncertainty CSV."""
    # Backward compatibility: legacy signature (path, result, peaks=None, method_label="")
    if result is None and peaks is None and not isinstance(file_path, (str, Path)):
        result = file_path
        file_path = None

    m = _unc_as_mapping(result)
    label = method_label or _method_label(m, default="")
    if rmse is None:
        rmse = m.get("rmse")
    if dof is None:
        dof = m.get("dof")

    per_peak = _rows_to_per_peak_stats(m, peaks or [])
    mlow = (label or "").lower()
    meta: Dict[str, Any] = {}
    if "bootstrap" in mlow:
        meta = {"backend": m.get("backend", ""), "n_boot": m.get("n_boot", "")}
    elif "bayesian" in mlow:
        meta = {
            "backend": "emcee",
            "n_draws": m.get("n_draws", ""),
            "ess": m.get("ess", ""),
            "rhat": m.get("rhat", ""),
        }
    rows = build_uncertainty_rows(
        str(file_path) if file_path else "",
        label,
        rmse,
        dof,
        per_peak,
        meta,
    )
    _write_uncertainty_csv(out_path, rows)
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
    """Compatibility wrapper producing a unified uncertainty TXT report."""
    if result is None and peaks is None and not isinstance(file_path, (str, Path)):
        result = file_path
        file_path = None

    m = _unc_as_mapping(result)
    label = method_label or _method_label(m, default="")
    per_peak = _rows_to_per_peak_stats(m, peaks or [])

    if isinstance(solver_meta, str):
        solver_line = solver_meta
    else:
        s = solver_meta or {}
        solver_line = "Solver: {solver}{loss}{weight}{f}{mfev}{rs}{jit}".format(
            solver=s.get("solver", "unknown"),
            loss=f", loss={s.get('loss')}" if s.get("loss") is not None else "",
            weight=f", weight={s.get('weight')}" if s.get("weight") is not None else "",
            f=f", f_scale={s.get('f_scale')}" if s.get("f_scale") is not None else "",
            mfev=f", maxfev={s.get('maxfev')}" if s.get("maxfev") is not None else "",
            rs=f", restarts={s.get('restarts')}" if s.get("restarts") is not None else "",
            jit=f", jitter_pct={s.get('jitter_pct')}" if s.get("jitter_pct") is not None else "",
        )
    if isinstance(baseline_meta, str):
        baseline_line = baseline_meta
    else:
        b = baseline_meta or {}
        baseline_line = "Baseline: uses_fit_range={uses} , lam={lam} , p={p} , niter={niter} , thresh={th}".format(
            uses=b.get("uses_fit_range", False),
            lam=b.get("lam"),
            p=b.get("p"),
            niter=b.get("niter"),
            th=b.get("thresh"),
        )
    if isinstance(perf_meta, str):
        perf_line = perf_meta
    else:
        pm = perf_meta or {}
        perf_line = "Performance: numba={numba}, gpu={gpu}, cache_baseline={cache}, seed_all={seed}, max_workers={mw}".format(
            numba=pm.get("numba"),
            gpu=pm.get("gpu"),
            cache=pm.get("cache_baseline"),
            seed=pm.get("seed_all"),
            mw=pm.get("max_workers"),
        )
    _write_uncertainty_txt(
        out_path,
        str(file_path),
        label,
        solver_line,
        baseline_line,
        perf_line,
        per_peak,
        z,
    )
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
    return export_uncertainty_txt(
        path,
        file_path=file_path,
        method_label=method_label,
        solver_meta=solver_meta,
        baseline_meta=baseline_meta,
        perf_meta=perf_meta,
        peaks=peaks,
        result=unc_res,
    )


