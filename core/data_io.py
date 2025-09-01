"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Mapping

import csv
import io
import math
import numpy as np
import re
from pathlib import Path
from math import isnan
import pandas as pd

from .uncertainty import UncertaintyResult

# 95% normal quantile used when p2_5/p97_5 are absent but stderr exists
_Z = 1.96


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

# --- BEGIN: Uncertainty helpers (stable public surface for UI/tests) ---

def _canonical_unc_label(label: Optional[str]) -> str:
    """Map many possible uncertainty method aliases to a canonical label."""
    s = str(label or "").strip().lower()
    # normalize spacing/symbols for JᵀJ variants
    s = s.replace("j^t j", "j^tj").replace("j^t  j", "j^tj")
    s = s.replace("jᵀ j", "jᵀj").replace("j ᵀ j", "jᵀj")

    asym_hits = (
        "asym" in s
        or "jtj" in s
        or "j^tj" in s
        or "jᵀj" in s
        or "gauss" in s
        or "hessian" in s
        or "linearized" in s
        or "curvature" in s
        or "cov" == s
        or "covariance" in s
        or "covmatrix" in s
    )
    boot_hits = (
        "boot" in s
        or "bootstrap" in s
        or "resample" in s
        or "resampling" in s
        or "resid" in s           # residual / residuals
        or "residual" in s
        or "percentile" in s
        or "perc" in s
    )
    bayes_hits = (
        "bayes" in s
        or "mcmc" in s
        or "emcee" in s
        or "pymc" in s
        or "numpyro" in s
        or "hmc" in s
        or "nuts" in s
        or "posterior" in s
        or "chain" in s
    )

    if asym_hits:
        return "Asymptotic (JᵀJ)"
    if boot_hits:
        return "Bootstrap (residual)"
    if bayes_hits:
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
    roots = ["stats", "parameters", "param_stats", "params"]
    rows = None
    for k in roots:
        if k in unc_map and unc_map[k] is not None:
            rows = unc_map[k]
            break
    if rows is None:
        return []

    # --- mapping-of-lists path (param -> {est:[...], sd:[...], ci_lo:[...], ...}) ---
    # Accepts legacy/alias param names like mu/x0/pos -> center, amp/amplitude -> height,
    # sigma/gamma/width -> fwhm, mix/mixing -> eta
    if isinstance(rows, Mapping):
        rows_map = {k: _as_mapping(v) for k, v in rows.items()}

        # alias sets
        aliases = {
            "center": {"center", "centre", "mu", "x0", "pos"},
            "height": {"height", "amp", "amplitude"},
            "fwhm":   {"fwhm", "width", "gamma", "sigma"},
            "eta":    {"eta", "mix", "mixing"},
        }

        # reverse index for quick lookup
        def _find_block(target: str) -> Mapping[str, Any]:
            keys = aliases[target]
            for k in rows_map.keys():
                kk = str(k).strip().lower()
                if kk in keys:
                    return rows_map[k]
            # tolerate pluralization
            for k in rows_map.keys():
                kk = str(k).strip().lower().rstrip("s")
                if kk in keys:
                    return rows_map[k]
            return {}

        blocks = {
            "center": _find_block("center"),
            "height": _find_block("height"),
            "fwhm":   _find_block("fwhm"),
            "eta":    _find_block("eta"),
        }

        def _vec_len(rec: Mapping[str, Any]) -> int:
            for key in ("est", "value", "mean", "median", "sd", "stderr", "sigma", "ci_lo", "ci_hi", "p2_5", "p97_5"):
                v = rec.get(key)
                if isinstance(v, (list, tuple, np.ndarray)):
                    return len(v)
            return 1 if rec else 0

        has_any = any(bool(b) for b in blocks.values())
        n_peaks = max((_vec_len(b) for b in blocks.values()), default=0) if has_any else 0
        if has_any and n_peaks == 0:
            n_peaks = 1

        if has_any:
            def pick(v, i):
                return (v[i] if isinstance(v, (list, tuple, np.ndarray)) and i < len(v) else v)

            out: List[Mapping[str, Any]] = []
            for i in range(n_peaks):
                row: Dict[str, Any] = {"index": i + 1}
                for pname in ("center", "height", "fwhm", "eta"):
                    rec = blocks[pname] or {}
                    est   = pick(rec.get("est")    or rec.get("value") or rec.get("mean")   or rec.get("median"), i)
                    sd    = pick(rec.get("sd")     or rec.get("stderr") or rec.get("sigma"), i)
                    lo    = pick(rec.get("ci_lo")  or rec.get("lo"), i)
                    hi    = pick(rec.get("ci_hi")  or rec.get("hi"), i)
                    p2_5  = pick(rec.get("p2_5")   or rec.get("p2.5")  or rec.get("q025")  or rec.get("q2_5"), i)
                    p97_5 = pick(rec.get("p97_5")  or rec.get("p97.5") or rec.get("q975")  or rec.get("q97_5"), i)

                    # synthesize CI if missing but SD present
                    if (
                        (lo is None or np.isnan(_to_float(lo))) and
                        (hi is None or np.isnan(_to_float(hi))) and
                        est is not None and sd is not None
                    ):
                        try:
                            e = float(est); s = float(sd)
                            lo, hi = e - _Z * s, e + _Z * s
                        except Exception:
                            pass

                    row[pname] = {
                        "est":   _to_float(est),
                        "sd":    _to_float(sd),
                        "ci_lo": _to_float(lo),
                        "ci_hi": _to_float(hi),
                        "p2_5":  _to_float(p2_5),
                        "p97_5": _to_float(p97_5),
                    }
                out.append(row)
            return out

        # p-indexed flat mapping: {'p0': {...}, 'p1': {...}, ...}
        if all(re.fullmatch(r"p\d+", str(k).strip().lower()) for k in rows_map.keys()):
            try:
                idx_map = {int(str(k).strip().lower()[1:]): _as_mapping(v) for k, v in rows_map.items()}
            except Exception:
                idx_map = {}
            if idx_map:
                n_params = 4  # center, height, fwhm, eta
                max_idx = max(idx_map.keys())
                n_peaks = max_idx // n_params + 1
                out: List[Mapping[str, Any]] = []
                for pk in range(n_peaks):
                    row: Dict[str, Any] = {"index": pk + 1}
                    for j, pname in enumerate(("center", "height", "fwhm", "eta")):
                        rec = idx_map.get(pk * n_params + j, {})
                        est   = rec.get("est")    or rec.get("value") or rec.get("mean")   or rec.get("median")
                        sd    = rec.get("sd")     or rec.get("stderr") or rec.get("sigma")
                        lo    = rec.get("ci_lo")  or rec.get("lo")
                        hi    = rec.get("ci_hi")  or rec.get("hi")
                        p2_5  = rec.get("p2_5")   or rec.get("p2.5")   or rec.get("q025")  or rec.get("q2_5")
                        p97_5 = rec.get("p97_5")  or rec.get("p97.5")  or rec.get("q975")  or rec.get("q97_5")
                        if (
                            (lo is None or np.isnan(_to_float(lo))) and
                            (hi is None or np.isnan(_to_float(hi))) and
                            est is not None and sd is not None
                        ):
                            try:
                                e = float(est); s = float(sd)
                                lo, hi = e - _Z * s, e + _Z * s
                            except Exception:
                                pass
                        row[pname] = {
                            "est":   _to_float(est),
                            "sd":    _to_float(sd),
                            "ci_lo": _to_float(lo),
                            "ci_hi": _to_float(hi),
                            "p2_5":  _to_float(p2_5),
                            "p97_5": _to_float(p97_5),
                        }
                    out.append(row)
                return out

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
            p2_5 = pick(rm, "p2_5", "p2.5", "q025", "q2_5")
            p97_5 = pick(rm, "p97_5", "p97.5", "q975", "q97_5")
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
    """Normalize arbitrary uncertainty payloads into a common mapping."""
    m = _as_mapping(unc)
    # Accept a wide range of potential keys describing the method/label
    raw_method = (
        m.get("label")
        or m.get("method_label")
        or m.get("method")
        or m.get("type")
        or m.get("mode")
        or m.get("name")
        or m.get("uncertainty")
        or m.get("uncertainty_method")
        or m.get("algorithm")
        or m.get("algo")
        or "unknown"
    )
    method = str(raw_method)
    canon = _canonical_unc_label(raw_method)

    # Pull band in a tolerant way
    band = None
    for key in ("band", "prediction_band", "ci_band"):
        if key in m and m[key] is not None:
            b = m[key]
            if isinstance(b, (list, tuple)) and len(b) >= 3:
                x, lo, hi = b[0], b[1], b[2]
                band = (np.asarray(x, float), np.asarray(lo, float), np.asarray(hi, float))
            break

    diag = _as_mapping(m.get("diagnostics"))

    out = {
        "label": canon,
        "method": method,
        "rmse": _to_float(m.get("rmse")),
        "dof": int(m.get("dof", m.get("d.o.f.", m.get("degrees_of_freedom", 0))) or 0),
        "backend": str(m.get("backend") or diag.get("backend") or m.get("engine") or ""),
        "n_draws": int(m.get("n_draws", m.get("samples", diag.get("n_draws", 0))) or 0),
        "n_boot": int(m.get("n_boot", m.get("bootstraps", diag.get("n_boot", diag.get("bootstraps", 0)))) or 0),
        "ess": _to_float(m.get("ess")),
        "rhat": _to_float(m.get("rhat")),
        "band": band,
        "stats": _extract_stats_table(m),
    }

    if out["label"] == "unknown":
        backend_s = str(out.get("backend", "")).lower()
        n_boot_i = int(out.get("n_boot") or 0)
        if n_boot_i > 0:
            out["label"] = "Bootstrap (residual)"
        elif any(k in backend_s for k in ("emcee", "pymc", "numpyro", "mcmc", "hmc", "nuts")):
            out["label"] = "Bayesian (MCMC)"
        elif any(k in backend_s for k in ("jtj", "j^tj", "jᵀj", "gauss", "hessian", "linearized", "cov")):
            out["label"] = "Asymptotic (JᵀJ)"

    return out


def _iter_param_rows(
    file_path: Union[str, Path, Any],
    unc_res: Any,
    *_,
) -> Iterable[Mapping[str, Any]]:
    """
    Yield CSV rows in the unified schema:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof,
      p2_5, p97_5, backend, n_draws, n_boot, ess, rhat
    """
    if isinstance(file_path, (str, Path)):
        fname = str(file_path)
        unc_norm = _normalize_unc_result(unc_res)
    else:  # backward-compat call: first arg is unc_res
        fname = ""
        unc_norm = _normalize_unc_result(file_path)
    canon_label = _canonical_unc_label(
        unc_norm.get("label") or unc_norm.get("method") or "unknown"
    )
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
                "method": canon_label,
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
    # Keep legacy label EXACT on the "Uncertainty method:" line
    canon_label = _canonical_unc_label(unc_norm.get("label") or unc_norm.get("method") or "unknown")
    if not canon_label:
        canon_label = "unknown"

    def fmt(x, nd=6):
        try:
            v = float(x)
            if not np.isfinite(v):
                return "n/a"
            return f"{v:.{nd}g}"
        except Exception:
            return "n/a"

    lines = [f"Uncertainty method: {canon_label}"]

    stats = unc_norm.get("stats", [])
    for i, row in enumerate(stats, start=1):

        def fmt_param(name: str, locked: bool):
            p = _as_mapping(row.get(name))
            est = p.get("est")
            sd = p.get("sd")
            lo = p.get("ci_lo")
            hi = p.get("ci_hi")
            # If SD missing but we have CI, estimate SD from CI width for display so we can emit ±
            if (sd is None or np.isnan(_to_float(sd))) and not (np.isnan(_to_float(lo)) or np.isnan(_to_float(hi))):
                try:
                    sd = float(hi - lo) / (2.0 * _Z)
                except Exception:
                    pass
            if locked:
                lines.append(f"  {name:<7}= {fmt(est)} (fixed)")
            else:
                if np.isnan(_to_float(sd)) and (np.isnan(_to_float(lo)) or np.isnan(_to_float(hi))):
                    # nothing reliable to show; keep n/a
                    lines.append(f"  {name:<7}= {fmt(est)} ± {fmt(sd,3)}")
                elif np.isnan(_to_float(lo)) or np.isnan(_to_float(hi)):
                    # have sd but not CI
                    lines.append(f"  {name:<7}= {fmt(est)} ± {fmt(sd,3)}")
                else:
                    # have CI; show both ± and CI
                    lines.append(f"  {name:<7}= {fmt(est)} ± {fmt(sd,3)}   (95% CI: [{fmt(lo)}, {fmt(hi)}])")

        # locks: same order as UI (center, height, fwhm, eta) default False if not provided
        lock_row = (_as_mapping(locks[i-1]) if i-1 < len(locks) else {})
        fmt_param("center", bool(lock_row.get("center", False)))
        fmt_param("height", False)
        fmt_param("fwhm",   bool(lock_row.get("fwhm", False)))
        fmt_param("eta",    bool(lock_row.get("eta", False)))

        # --- NEW: p-indexed legacy summary lines (satisfies tests looking for "p0:", ... and "±") ---
        # p0->center, p1->height, p2->fwhm, p3->eta
        def pick_est_sd(name: str) -> Tuple[str, str]:
            p = _as_mapping(row.get(name))
            est = p.get("est")
            sd = p.get("sd")
            lo = p.get("ci_lo")
            hi = p.get("ci_hi")
            if (sd is None or np.isnan(_to_float(sd))) and not (np.isnan(_to_float(lo)) or np.isnan(_to_float(hi))):
                try:
                    sd = float(hi - lo) / (2.0 * _Z)
                except Exception:
                    pass
            return fmt(est), fmt(sd, 3)

        c_est, c_sd = pick_est_sd("center")
        h_est, h_sd = pick_est_sd("height")
        w_est, w_sd = pick_est_sd("fwhm")
        e_est, e_sd = pick_est_sd("eta")
        lines.append(f"  p0: {c_est} ± {c_sd}")
        lines.append(f"  p1: {h_est} ± {h_sd}")
        lines.append(f"  p2: {w_est} ± {w_sd}")
        lines.append(f"  p3: {e_est} ± {e_sd}")
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


def iter_uncertainty_rows(
    file_path: Union[str, Path],
    unc_norm: Mapping[str, Any],
) -> List[dict]:
    """
    Build long-format rows with guaranteed canonical labels and p2_5/p97_5.
    Columns: file, peak, param, value, stderr, p2_5, p97_5, method, rmse, dof.
    If quantiles are missing but stderr is present, synthesize via ±1.96*stderr.
    """
    fname = Path(file_path).name if file_path else ""
    method = str(unc_norm.get("label") or unc_norm.get("method") or "unknown")
    rmse = unc_norm.get("rmse", float("nan"))
    dof = unc_norm.get("dof", float("nan"))

    rows: List[dict] = []
    stats = unc_norm.get("stats") or []
    for i, row in enumerate(stats, start=1):
        for param in ("center", "height", "fwhm", "eta"):
            blk = (row or {}).get(param) or {}
            est = blk.get("est")
            sd = blk.get("sd")
            qlo = blk.get("p2_5")
            qhi = blk.get("p97_5")
            if (qlo is None or qhi is None) and (est is not None and sd is not None):
                try:
                    e = float(est)
                    s = float(sd)
                    qlo, qhi = e - _Z * s, e + _Z * s
                except Exception:
                    pass
            rows.append(
                {
                    "file": fname,
                    "peak": i,
                    "param": param,
                    "value": est,
                    "stderr": sd,
                    "p2_5": qlo,
                    "p97_5": qhi,
                    "method": method,
                    "rmse": rmse,
                    "dof": dof,
                }
            )
    return rows


def write_uncertainty_csv_legacy(
    base_path: Union[str, Path],
    file_path: Union[str, Path],
    unc_norm: Mapping[str, Any],
) -> Path:
    """
    Write <base>_uncertainty.csv (long format).
    """
    base = Path(base_path).with_suffix("")
    out_csv = base.with_name(base.name + "_uncertainty.csv")
    rows = iter_uncertainty_rows(file_path, unc_norm)
    header = [
        "file","peak","param","value","stderr","p2_5","p97_5","method","rmse","dof"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv


def write_uncertainty_csvs(
    base_path: Path,
    file_path: Union[str, Path],
    unc_norm: Mapping[str, Any],
    *,
    write_wide: bool = True,
):
    """
    Write per-file uncertainty CSVs:
      - always: <base>_uncertainty.csv   (long)
      - optional: <base>_uncertainty_wide.csv (wide)
    """
    base_path = Path(base_path).with_suffix("")
    long_written = write_uncertainty_csv_legacy(base_path, file_path, unc_norm)

    wide_written = None
    if write_wide:
        rows = iter_uncertainty_rows(file_path, unc_norm)
        wide_path = base_path.with_name(base_path.name + "_uncertainty_wide.csv")
        header = [
            "file","peak","method","rmse","dof",
            "center","center_stderr","center_ci_lo","center_ci_hi",
            "height","height_stderr","height_ci_lo","height_ci_hi",
            "fwhm","fwhm_stderr","fwhm_ci_lo","fwhm_ci_hi",
            "eta","eta_stderr","eta_ci_lo","eta_ci_hi",
        ]
        by_peak: dict[int, dict[str, Any]] = {}
        for r in rows:
            pk = r["peak"]
            d = by_peak.setdefault(
                pk,
                {
                    "file": r["file"],
                    "peak": pk,
                    "method": r["method"],
                    "rmse": r["rmse"],
                    "dof": r["dof"],
                },
            )
            p = r["param"]
            d[f"{p}"] = r["value"]
            d[f"{p}_stderr"] = r["stderr"]
            qlo, qhi = r.get("p2_5"), r.get("p97_5")
            if (qlo is None or qhi is None) and (r.get("value") is not None and r.get("stderr") is not None):
                try:
                    e = float(r["value"])
                    s = float(r["stderr"])
                    qlo, qhi = e - _Z * s, e + _Z * s
                except Exception:
                    pass
            d[f"{p}_ci_lo"] = qlo
            d[f"{p}_ci_hi"] = qhi

        with wide_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            w.writeheader()
            for pk in sorted(by_peak):
                w.writerow(by_peak[pk])
        wide_written = wide_path

    return str(long_written), (str(wide_written) if wide_written else None)


def write_batch_uncertainty_long(
    out_dir: Union[str, Path],
    rows: List[dict],
) -> tuple[Path, Path]:
    """
    Write aggregated rows to:
      - batch_uncertainty_long.csv (preferred)
      - batch_uncertainty.csv      (legacy-compatible mirror)
    """
    out_dir = Path(out_dir)
    header = ["file","peak","param","value","stderr","p2_5","p97_5","method","rmse","dof"]

    def _write(path: Path):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    long_path = out_dir / "batch_uncertainty_long.csv"
    legacy_path = out_dir / "batch_uncertainty.csv"
    _write(long_path)
    _write(legacy_path)
    return long_path, legacy_path

# --- END: add wide exporter ---

# --- END: Uncertainty helpers ---

# Backwards-compatible wrappers
canonical_unc_label = _canonical_unc_label
normalize_unc_result = _normalize_unc_result


# Note: keep _ensure_result available if used elsewhere
def _ensure_result(unc: Any) -> UncertaintyResult:
    """Coerce *unc* into an UncertaintyResult, tolerating legacy shapes."""
    if isinstance(unc, UncertaintyResult):
        return unc
    m = _as_mapping(unc)
    method = str(m.get("type") or m.get("method") or "unknown")
    label = _canonical_unc_label(m.get("label") or m.get("method_label") or m.get("method") or method)
    stats = _as_mapping(m.get("param_stats") or m.get("parameters") or m.get("params") or m.get("stats"))
    diag = _as_mapping(m.get("diagnostics"))
    band = m.get("band")
    return UncertaintyResult(method=method, label=label, stats=stats, diagnostics=diag, band=band)


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
    # Back-compat: the *singular* API writes a single-row "wide" CSV with
    # p-indexed columns (p0, p0_sd, ...).  Legacy tests expect these names.
    unc = _normalize_unc_result(unc_res)
    res_obj = _ensure_result(unc_res)
    fname = str(file_path)
    row: Dict[str, Any] = {
        "file": fname,
        "method": unc.get("label", "unknown"),
        "rmse": _to_float(unc.get("rmse")),
        "dof": _to_float(unc.get("dof")),
        "backend": unc.get("backend", ""),
        "n_draws": _to_float(unc.get("n_draws")),
        "n_boot": _to_float(unc.get("n_boot")),
        "ess": _to_float(unc.get("ess")),
        "rhat": _to_float(unc.get("rhat")),
    }

    header = [
        "file",
        "method",
        "rmse",
        "dof",
        "backend",
        "n_draws",
        "n_boot",
        "ess",
        "rhat",
    ]

    stats_map = _as_mapping(getattr(res_obj, "stats", {}))
    for i, (name, st) in enumerate(stats_map.items()):
        p = _as_mapping(st)
        row.update(
            {
                name: _to_float(p.get("est") or p.get("mean") or p.get("value")),
                f"{name}_sd": _to_float(p.get("sd") or p.get("stderr") or p.get("sigma")),
                f"{name}_ci_lo": _to_float(p.get("p2.5") or p.get("ci_lo")),
                f"{name}_ci_hi": _to_float(p.get("p97.5") or p.get("ci_hi")),
                f"{name}_p2_5": _to_float(p.get("p2_5")),
                f"{name}_p97_5": _to_float(p.get("p97_5")),
            }
        )
        header.extend(
            [
                name,
                f"{name}_sd",
                f"{name}_ci_lo",
                f"{name}_ci_hi",
                f"{name}_p2_5",
                f"{name}_p97_5",
            ]
        )

    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerow(row)


def write_uncertainty_txt(
    out_path: Union[str, Path],
    unc_norm: Mapping[str, Any] | Any,
    *,
    peaks=None,
    method_label: str = "",
    file_path: Union[str, Path] = "",
    solver_meta: Mapping[str, Any] | None = None,
    baseline_meta: Mapping[str, Any] | None = None,
    perf_meta: Mapping[str, Any] | None = None,
    locks: Sequence[Mapping[str, bool]] | None = None,
    **_: Any,
) -> None:
    """
    Public wrapper for writing a human-readable uncertainty report.
    Delegate to existing internal writer if available.
    """
    try:
        unc = normalize_unc_result(unc_norm)
        return _write_unc_txt(
            out_path,
            file_path,
            unc,
            solver_meta or {},
            baseline_meta or {},
            perf_meta or {},
            locks or [],
        )
    except NameError:
        unc = normalize_unc_result(unc_norm)
        label = unc.get("label", "unknown")
        with Path(out_path).open("w", encoding="utf-8") as fh:
            fh.write(f"Uncertainty method: {label}\n")
            fh.write(f"File: {Path(file_path).name if file_path else ''}\n\n")
            for i, row in enumerate(unc.get('stats', []), start=1):
                c = (row.get('center', {}) or {})
                h = (row.get('height', {}) or {})
                w = (row.get('fwhm', {}) or {})
                e = (row.get('eta', {}) or {})

                def fmt(blk):
                    est = blk.get('est')
                    sd = blk.get('sd')
                    return f"{est if est is not None else 'n/a'} ± {sd if sd is not None else 'n/a'}"

                fh.write(
                    f"Peak {i}: center={fmt(c)} | height={fmt(h)} | FWHM={fmt(w)} | eta={fmt(e)}\n"
                )


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

