"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
        row.extend(comps_arr[:, idx] if comps else [])
        row.append(y_target_sub[idx])
        row.append(y_fit_sub[idx])
        row.extend(comps_arr[:, idx] if comps else [])
        writer.writerow(row)
    return buf.getvalue()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` without introducing extra blank lines."""

    with path.open("w", newline="", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


