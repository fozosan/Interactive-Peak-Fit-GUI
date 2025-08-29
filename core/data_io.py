"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import csv
import io
import re
from pathlib import Path

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


def _normalize_band(result):
    """Return ``(x, lo, hi)`` arrays or ``None``.

    Accepts ``UncertaintyResult`` instances or dict-like structures containing a
    band description. The function avoids evaluating numpy arrays in boolean
    context and validates shapes before returning normalized arrays.
    """

    if result is None:
        return None

    band = getattr(result, "band", None)
    if band is None:
        band = getattr(result, "prediction_band", None)

    if band is None and isinstance(result, dict):
        band = result.get("band")
        if band is None:
            band = result.get("prediction_band")
        if band is None:
            band = result.get("ci_band")

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


def write_uncertainty_csv(path: str | Path, unc_res, peaks=None, method_label: str = "") -> None:
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
        return

    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["peak", "param", "est", "sd", "p2_5", "p97_5", "method"],
            lineterminator="\n",
        )
        w.writeheader()
        for row in _iter_param_rows(unc_res, peaks, method_label):
            w.writerow(row)


def write_uncertainty_txt(path: str | Path, unc_res, peaks=None, method_label: str = "") -> None:
    if peaks is None:
        res = _ensure_result(unc_res)
        lines = [f"Method: {res.method_label}"]
        for name, stats in res.param_stats.items():
            est = stats.get("est")
            sd = stats.get("sd")
            line = f"{name}: {est:.6g} ± {sd:.6g}"
            if "p2.5" in stats and "p97.5" in stats:
                line += f"   [2.5%: {stats['p2.5']:.6g}, 97.5%: {stats['p97.5']:.6g}]"
            lines.append(line)
        text = "\n".join(lines) + "\n"
        Path(path).write_text(text, encoding="utf-8")
        return

    stats = getattr(unc_res, "stats", None)
    if stats is None and isinstance(unc_res, dict):
        stats = unc_res.get("stats")
        if stats is None:
            stats = unc_res.get("parameters")
        if stats is None:
            stats = unc_res.get("param_stats")
    lines = [f"Uncertainty: {method_label}"]
    if not stats:
        lines.append("No parameter statistics available.")
    else:
        def fmt(d):
            est = d.get("est")
            sd = d.get("sd")
            if est is None or sd is None:
                return "n/a"
            return f"{est:.6g} ± {sd:.3g}"

        for i, _ in enumerate(peaks, 1):
            s = stats.get(i) or stats.get(str(i)) or {}
            c = fmt(s.get("center", {}))
            h = fmt(s.get("height", {}))
            w = fmt(s.get("fwhm", {}))
            lines.append(f"Peak {i}: center={c} | height={h} | FWHM={w}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


