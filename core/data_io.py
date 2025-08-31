"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import csv
import io
import re
from pathlib import Path

import math
from math import isnan
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

# --- BEGIN: Uncertainty helpers (stable public surface for UI/tests) ---

def _canonical_unc_label(label: Optional[str]) -> str:
    """
    Map any label/method/type to a human-friendly canonical name.
    Returns one of: "Asymptotic (JᵀJ)", "Bootstrap (residual)", "Bayesian (MCMC)", "asymptotic", "bootstrap", "bayesian".
    """
    if not label:
        return "unknown"
    s = str(label).strip().lower()
    if "asym" in s or "j" in s and "j" in s:  # tolerate variants like "Asymptotic (J^T J)"
        return "Asymptotic (JᵀJ)"
    if "boot" in s:
        return "Bootstrap (residual)"
    if "bayes" in s or "mcmc" in s:
        return "Bayesian (MCMC)"
    return "unknown"


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """Return obj as a read-only mapping interface."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    # list/tuple results (common in older bootstrap paths) -> wrap under a 'stats' key
    if isinstance(obj, (list, tuple)):
        return {"stats": obj}
    return {"value": obj}


def _extract_stats_table(unc_map: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    roots = ["stats", "parameters", "param_stats"]
    rows = None
    for k in roots:
        if k in unc_map and unc_map[k] is not None:
            rows = unc_map[k]
            break
    if rows is None:
        return []

    # Helper pickers
    def pick(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def norm_param_name(s: Any) -> Optional[str]:
        if not isinstance(s, str):
            return None
        ss = s.strip().lower()
        if ss in ("center", "centre", "mu", "x0", "pos"):
            return "center"
        if ss in ("height", "amp", "amplitude"):
            return "height"
        if ss in ("fwhm", "width", "gamma", "sigma"):
            return "fwhm"
        if ss in ("eta", "mix", "mixing"):
            return "eta"
        return None

    # --- Long-format path: rows like {'peak', 'param', 'est'/ 'value', 'sd'/'stderr', ...}
    if (
        isinstance(rows, (list, tuple))
        and rows
        and isinstance(rows[0], Mapping)
        and ("param" in rows[0] or "name" in rows[0])
    ):
        from collections import defaultdict

        by_peak: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)

        for r in rows:
            rm = _as_mapping(r)
            # Identify peak index
            idx = pick(rm, "peak", "index", "k", "i")
            try:
                pk_idx = int(idx) if idx is not None else None
            except Exception:
                pk_idx = None
            if pk_idx is None:
                continue

            # Param name -> canonical
            rawp = pick(rm, "param", "name")
            pname = norm_param_name(rawp)
            if not pname:
                continue

            est = pick(rm, "est", "value", "mean", "median")
            sd = pick(rm, "sd", "stderr", "std", "stdev")
            p2_5 = pick(rm, "p2_5", "q025", "q2_5")
            p97_5 = pick(rm, "p97_5", "q975", "q97_5")
            ci_lo = pick(rm, "ci_lo")
            ci_hi = pick(rm, "ci_hi")

            # Synthesize CI from sd if missing
            if (ci_lo is None or ci_hi is None) and est is not None and sd is not None:
                try:
                    e = float(est)
                    s = float(sd)
                    ci_lo, ci_hi = e - _Z * s, e + _Z * s
                except Exception:
                    pass

            by_peak[pk_idx][pname] = {
                "est": _to_float(est),
                "sd": _to_float(sd),
                "ci_lo": _to_float(ci_lo),
                "ci_hi": _to_float(ci_hi),
                "p2_5": _to_float(p2_5),
                "p97_5": _to_float(p97_5),
            }

        out = []
        for pk in sorted(by_peak.keys()):
            block = by_peak[pk]
            out.append(
                {
                    "index": int(pk),
                    "center": block.get(
                        "center",
                        {
                            "est": float("nan"),
                            "sd": float("nan"),
                            "ci_lo": float("nan"),
                            "ci_hi": float("nan"),
                            "p2_5": float("nan"),
                            "p97_5": float("nan"),
                        },
                    ),
                    "height": block.get(
                        "height",
                        {
                            "est": float("nan"),
                            "sd": float("nan"),
                            "ci_lo": float("nan"),
                            "ci_hi": float("nan"),
                            "p2_5": float("nan"),
                            "p97_5": float("nan"),
                        },
                    ),
                    "fwhm": block.get(
                        "fwhm",
                        {
                            "est": float("nan"),
                            "sd": float("nan"),
                            "ci_lo": float("nan"),
                            "ci_hi": float("nan"),
                            "p2_5": float("nan"),
                            "p97_5": float("nan"),
                        },
                    ),
                    "eta": block.get(
                        "eta",
                        {
                            "est": float("nan"),
                            "sd": float("nan"),
                            "ci_lo": float("nan"),
                            "ci_hi": float("nan"),
                            "p2_5": float("nan"),
                            "p97_5": float("nan"),
                        },
                    ),
                }
            )
        return out

    # --- Existing tolerant per-peak/flat path (keep your current logic) ---
    norm_rows: List[Mapping[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        rmap = _as_mapping(row)
        def param_block(base: str) -> Mapping[str, float]:
            block = _as_mapping(rmap.get(base))
            if block:
                est = block.get("est", block.get("value", np.nan))
                sd = block.get("sd", block.get("stderr", np.nan))
                lo = block.get("ci_lo", block.get("lo", np.nan))
                hi = block.get("ci_hi", block.get("hi", np.nan))
                p2_5 = block.get("p2_5", np.nan)
                p97_5 = block.get("p97_5", np.nan)
            else:
                est = rmap.get(f"{base}_est", rmap.get(base, np.nan))
                sd = rmap.get(f"{base}_sd", rmap.get(f"{base}_stderr", np.nan))
                lo = rmap.get(f"{base}_ci_lo", np.nan)
                hi = rmap.get(f"{base}_ci_hi", np.nan)
                p2_5 = rmap.get(f"{base}_p2_5", np.nan)
                p97_5 = rmap.get(f"{base}_p97_5", np.nan)
            # synthesize CI if missing but sd present
            if (
                (lo is None or np.isnan(_to_float(lo)))
                and (hi is None or np.isnan(_to_float(hi)))
                and est is not None
                and sd is not None
            ):
                try:
                    e = float(est)
                    s = float(sd)
                    lo, hi = e - _Z * s, e + _Z * s
                except Exception:
                    pass
            return {
                "est": _to_float(est),
                "sd": _to_float(sd),
                "ci_lo": _to_float(lo),
                "ci_hi": _to_float(hi),
                "p2_5": _to_float(p2_5),
                "p97_5": _to_float(p97_5),
            }

        norm_rows.append(
            {
                "index": int(rmap.get("index", rmap.get("peak", i)) or i),
                "center": param_block("center"),
                "height": param_block("height"),
                "fwhm": param_block("fwhm"),
                "eta": param_block("eta"),
            }
        )
    return norm_rows


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _normalize_unc_result(unc: Any) -> Mapping[str, Any]:
    """
    Normalize any uncertainty result into a dict with fields:
      label (canonical string), rmse (float), dof (int), backend (str),
      n_draws (int), n_boot (int), ess (float), rhat (float),
      band (optional 3-tuple: (x, lo, hi)),
      stats: List[ per-peak mapping as described in _extract_stats_table() ].
    Unknown fields may be absent; missing numeric values are np.nan.
    """
    m = _as_mapping(unc)
    label = m.get("label") or m.get("method_label") or m.get("method") or m.get("type")
    canon = _canonical_unc_label(label)

    # Pull band in a tolerant way
    band = None
    for key in ("band", "prediction_band", "ci_band"):
        if key in m and m[key] is not None:
            b = m[key]
            if isinstance(b, (list, tuple)) and len(b) >= 3:
                x, lo, hi = b[0], b[1], b[2]
                band = (np.asarray(x, float), np.asarray(lo, float), np.asarray(hi, float))
            break

    out = {
        "label": canon,
        "rmse": _to_float(m.get("rmse")),
        "dof": int(m.get("dof", m.get("d.o.f.", m.get("degrees_of_freedom", 0))) or 0),
        "backend": str(m.get("backend", m.get("engine", ""))) or "",
        "n_draws": int(m.get("n_draws", m.get("samples", 0)) or 0),
        "n_boot": int(m.get("n_boot", m.get("bootstraps", 0)) or 0),
        "ess": _to_float(m.get("ess")),
        "rhat": _to_float(m.get("rhat")),
        "band": band,
        "stats": _extract_stats_table(m),
    }
    return out


def _iter_param_rows(
    file_path: Union[str, "Path", Any],
    unc_norm: Any,
    *_: Any,
) -> Iterable[Mapping[str, Any]]:
    """
    Yield CSV rows in the unified schema:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof,
      p2_5, p97_5, backend, n_draws, n_boot, ess, rhat
    """
    if not isinstance(file_path, (str, Path)) or isinstance(unc_norm, (list, tuple, Mapping)) and not isinstance(file_path, (str, Path)):
        unc_norm = _normalize_unc_result(file_path)
        fname = ""
    else:
        unc_norm = _normalize_unc_result(unc_norm)
        fname = str(file_path)
    label = unc_norm.get("label", "unknown")
    rmse  = _to_float(unc_norm.get("rmse"))
    dof   = int(unc_norm.get("dof", 0))
    backend = unc_norm.get("backend", "")
    n_draws = int(unc_norm.get("n_draws", 0))
    n_boot  = int(unc_norm.get("n_boot", 0))
    ess     = _to_float(unc_norm.get("ess"))
    rhat    = _to_float(unc_norm.get("rhat"))

    for row in unc_norm.get("stats", []):
        peak_idx = int(row.get("index", 0))
        for pname in ("center","height","fwhm","eta"):
            p = _as_mapping(row.get(pname))
            yield {
                "file": fname,
                "peak": peak_idx,
                "param": pname,
                "value": _to_float(p.get("est")),
                "stderr": _to_float(p.get("sd")),
                "ci_lo": _to_float(p.get("ci_lo")),
                "ci_hi": _to_float(p.get("ci_hi")),
                "method": label.lower().split()[0],  # asymptotic/bootstrap/bayesian/unknown
                "rmse": rmse,
                "dof": dof,
                "p2_5": _to_float(p.get("p2_5")),
                "p97_5": _to_float(p.get("p97_5")),
                "backend": backend,
                "n_draws": n_draws,
                "n_boot": n_boot,
                "ess": ess,
                "rhat": rhat,
            }


def _format_unc_text(
    file_path: Union[str, "Path"],
    unc_norm: Mapping[str, Any],
    solver_meta: Mapping[str, Any],
    baseline_meta: Mapping[str, Any],
    perf_meta: Mapping[str, Any],
    locks: Sequence[Mapping[str, bool]],
) -> str:
    """
    Return v2.7-style human-readable text with ± and 95% CI, marking (fixed) when locked.
    """
    raw_label = str(unc_norm.get("label", "unknown"))
    if raw_label.startswith("Asymptotic"):
        nice_label = "Asymptotic (95% CI, z=1.96)"
    elif raw_label.startswith("Bootstrap"):
        nice_label = "Bootstrap (95% CI via percentiles)"
    elif raw_label.startswith("Bayesian"):
        nice_label = "Bayesian (95% credible interval)"
    else:
        nice_label = raw_label

    def fmt(x, nd=6):
        try:
            v = float(x)
            if not np.isfinite(v):
                return "n/a"
            return f"{v:.{nd}g}"
        except Exception:
            return "n/a"

    fname = str(file_path)
    lines = []
    lines.append(f"File: {fname}")
    lines.append(f"Uncertainty method: {nice_label}")
    lines.append("Solver: " + ", ".join(f"{k}={v}" for k,v in solver_meta.items()))
    lines.append("Baseline: " + ", ".join(f"{k}={v}" for k,v in baseline_meta.items()))
    lines.append("Performance: " + ", ".join(f"{k}={v}" for k,v in perf_meta.items()))
    lines.append("Peaks:")

    stats = unc_norm.get("stats", [])
    for i, row in enumerate(stats, start=1):
        lock = locks[i-1] if i-1 < len(locks) else {"center": False, "fwhm": False, "eta": False}
        lines.append(f"Peak {i}")
        def fmt_param(name: str, locked: bool):
            p = _as_mapping(row.get(name))
            est = p.get("est")
            sd = p.get("sd")
            lo = p.get("ci_lo")
            hi = p.get("ci_hi")
            if locked:
                lines.append(f"  {name:<7}= {fmt(est)} (fixed)")
            else:
                if not (np.isnan(_to_float(lo)) or np.isnan(_to_float(hi))):
                    lines.append(
                        f"  {name:<7}= {fmt(est)} ± {fmt(sd,3)}   (95% CI: [{fmt(lo)}, {fmt(hi)}])"
                    )
                else:
                    lines.append(f"  {name:<7}= {fmt(est)} ± {fmt(sd,3)}")

        fmt_param("center", lock.get("center", False))
        fmt_param("height", False)
        fmt_param("fwhm",   lock.get("fwhm", False))
        fmt_param("eta",    lock.get("eta", False))
    return "\n".join(lines)


def _write_unc_csv(csv_path: Union[str, "Path"], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows with no blank lines and stable header ordering."""
    csv_path = Path(csv_path)
    header = [
        "file","peak","param","value","stderr","ci_lo","ci_hi",
        "method","rmse","dof","p2_5","p97_5","backend","n_draws","n_boot","ess","rhat"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _write_unc_txt(
    txt_path: Union[str, "Path"],
    file_path: Union[str, "Path"],
    unc_norm: Mapping[str, Any],
    solver_meta: Mapping[str, Any],
    baseline_meta: Mapping[str, Any],
    perf_meta: Mapping[str, Any],
    locks: Sequence[Mapping[str, bool]],
) -> None:
    """Write the human-readable TXT."""
    txt_path = Path(txt_path)
    txt = _format_unc_text(file_path, unc_norm, solver_meta, baseline_meta, perf_meta, locks)
    txt_path.write_text(txt, encoding="utf-8")

# --- BEGIN: add wide exporter next to existing long-format helpers ---

def _iter_peak_rows_wide(
    file_path: Union[str, "Path"],
    unc_norm: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    """
    Yield 'wide' per-peak rows:
      file, peak, method, rmse, dof, backend, n_draws, n_boot, ess, rhat,
      center, center_stderr, center_ci_lo, center_ci_hi, center_p2_5, center_p97_5,
      height, height_stderr, height_ci_lo, height_ci_hi, height_p2_5, height_p97_5,
      fwhm,   fwhm_stderr,   fwhm_ci_lo,   fwhm_ci_hi,   fwhm_p2_5,   fwhm_p97_5,
      eta,    eta_stderr,    eta_ci_lo,    eta_ci_hi,    eta_p2_5,    eta_p97_5
    """
    fname   = str(file_path)
    label   = unc_norm.get("label", "unknown")
    method  = label.lower().split()[0]  # asymptotic/bootstrap/bayesian/unknown
    rmse    = _to_float(unc_norm.get("rmse"))
    dof     = int(unc_norm.get("dof", 0))
    backend = unc_norm.get("backend", "")
    n_draws = int(unc_norm.get("n_draws", 0))
    n_boot  = int(unc_norm.get("n_boot", 0))
    ess     = _to_float(unc_norm.get("ess"))
    rhat    = _to_float(unc_norm.get("rhat"))

    for row in unc_norm.get("stats", []):
        peak_idx = int(row.get("index", 0))

        def pick(name: str) -> Mapping[str, float]:
            pm = _as_mapping(row.get(name))
            return {
                name: _to_float(pm.get("est")),
                f"{name}_stderr": _to_float(pm.get("sd")),
                f"{name}_ci_lo": _to_float(pm.get("ci_lo")),
                f"{name}_ci_hi": _to_float(pm.get("ci_hi")),
                f"{name}_p2_5": _to_float(pm.get("p2_5")),
                f"{name}_p97_5": _to_float(pm.get("p97_5")),
            }

        out = {
            "file": fname,
            "peak": peak_idx,
            "method": method,
            "rmse": rmse,
            "dof": dof,
            "backend": backend,
            "n_draws": n_draws,
            "n_boot": n_boot,
            "ess": ess,
            "rhat": rhat,
        }
        for pname in ("center", "height", "fwhm", "eta"):
            out.update(pick(pname))
        yield out


def _write_unc_csv_wide(csv_path: Union[str, "Path"], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write wide per-peak rows (legacy-friendly)."""
    csv_path = Path(csv_path)
    header = [
        "file","peak","method","rmse","dof","backend","n_draws","n_boot","ess","rhat",
        "center","center_stderr","center_ci_lo","center_ci_hi","center_p2_5","center_p97_5",
        "height","height_stderr","height_ci_lo","height_ci_hi","height_p2_5","height_p97_5",
        "fwhm","fwhm_stderr","fwhm_ci_lo","fwhm_ci_hi","fwhm_p2_5","fwhm_p97_5",
        "eta","eta_stderr","eta_ci_lo","eta_ci_hi","eta_p2_5","eta_p97_5",
    ]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_uncertainty_csvs(
    base_path: Union[str, "Path"],
    file_path: Union[str, "Path"],
    unc_norm: Mapping[str, Any],
    *,
    write_wide: bool = False,
) -> Tuple["Path", "Path"]:
    """
    Convenience: write long CSV to <base>_uncertainty.csv.
    If write_wide=True, also write <base>_uncertainty_wide.csv.
    Returns (long_csv_path, wide_csv_path_or_None).
    """
    base = Path(base_path).with_suffix("")
    long_csv = base.with_name(base.name + "_uncertainty.csv")
    rows_long = list(_iter_param_rows(file_path, unc_norm))
    _write_unc_csv(long_csv, rows_long)

    wide_csv = None
    if write_wide:
        wide_csv = base.with_name(base.name + "_uncertainty_wide.csv")
        rows_wide = list(_iter_peak_rows_wide(file_path, unc_norm))
        _write_unc_csv_wide(wide_csv, rows_wide)

    return long_csv, wide_csv

# --- END: add wide exporter ---

# --- END: Uncertainty helpers ---

# Backwards-compatible wrappers
canonical_unc_label = _canonical_unc_label
normalize_unc_result = _normalize_unc_result


def write_uncertainty_csv(
    path: Union[str, Path],
    unc_res: Any,
    peaks: Any | None = None,
    method_label: str = "",
    rmse: float | None = None,
    dof: float | None = None,
    file_path: Union[str, Path] = "",
    **_: Any,
) -> None:
    unc = _normalize_unc_result(unc_res)
    rows = list(_iter_peak_rows_wide(file_path, unc))
    _write_unc_csv_wide(path, rows)


def write_uncertainty_txt(
    path: Union[str, Path],
    unc_res: Any,
    peaks: Any | None = None,
    method_label: str = "",
    file_path: Union[str, Path] = "",
    solver_meta: Mapping[str, Any] | None = None,
    baseline_meta: Mapping[str, Any] | None = None,
    perf_meta: Mapping[str, Any] | None = None,
    locks: Sequence[Mapping[str, bool]] | None = None,
    **_: Any,
) -> None:
    unc = _normalize_unc_result(unc_res)
    _write_unc_txt(path, file_path, unc, solver_meta or {}, baseline_meta or {}, perf_meta or {}, locks or [])


def export_uncertainty_pair(
    out_csv: Union[str, Path],
    out_txt: Union[str, Path],
    unc_res: Any,
    file_path: Union[str, Path] = "",
    solver_meta: Mapping[str, Any] | None = None,
    baseline_meta: Mapping[str, Any] | None = None,
    perf_meta: Mapping[str, Any] | None = None,
    locks: Sequence[Mapping[str, bool]] | None = None,
) -> None:
    unc = _normalize_unc_result(unc_res)
    rows = list(_iter_param_rows(file_path, unc))
    _write_unc_csv(out_csv, rows)
    _write_unc_txt(out_txt, file_path, unc, solver_meta or {}, baseline_meta or {}, perf_meta or {}, locks or [])

