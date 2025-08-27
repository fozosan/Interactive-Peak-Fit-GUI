"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import csv
import io
import re
import numpy as np


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
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    return buf.getvalue()


def build_trace_table(
    x: np.ndarray,
    y_raw: np.ndarray,
    baseline: np.ndarray | None,
    peaks: Iterable,
    mode: str = "add",
) -> str:
    """Return a CSV trace table for the current fit.

    Parameters
    ----------
    x, y_raw:
        Input data arrays.
    baseline:
        Baseline array or ``None``.
    peaks:
        Iterable of peak-like objects. Each contributes its own column.
    mode:
        ``"add"`` or ``"subtract"`` — controls how the fitted target and
        model columns are formed.
    """

    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    base = np.asarray(baseline, dtype=float) if baseline is not None else 0.0

    # per-peak contributions
    comps = []
    from .models import pv_sum  # local import to avoid cycles

    for p in peaks:
        comps.append(pv_sum(x, [p]))

    comps_arr = np.vstack(comps) if comps else np.empty((0, x.size))
    model = comps_arr.sum(axis=0) if comps else np.zeros_like(x)

    if mode == "add":
        y_target = y_raw
        y_fit = model + base
    elif mode == "subtract":
        y_target = y_raw - base
        y_fit = model
    else:  # pragma: no cover - unknown mode
        raise ValueError("unknown mode")

    headers = ["x", "y_raw", "baseline", "y_target", "y_fit"] + [
        f"peak{i+1}" for i in range(comps_arr.shape[0])
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for idx in range(x.size):
        row = [
            x[idx],
            y_raw[idx],
            base[idx] if baseline is not None else 0.0,
            y_target[idx],
            y_fit[idx],
        ]
        row.extend(comps_arr[:, idx])
        writer.writerow(row)
    return buf.getvalue()

