"""Data I/O helpers for Peakfit 3.x.

This module provides functions to load spectral data and build export
artifacts. Implementations follow the Peakfit 3.x blueprint.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import csv
import io
import numpy as np


def _detect_delimiter(sample: str) -> str | None:
    """Heuristically guess a delimiter from *sample*.

    Returns a delimiter character or ``None`` to indicate generic whitespace.
    """

    for delim in (",", "\t", ";"):
        if delim in sample:
            return delim
    return None


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load two-column numeric data from ``path``.

    Comments beginning with ``#`` and optional header lines are tolerated.
    The delimiter is autodetected among comma, tab, semicolon and whitespace.
    """

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            delim = _detect_delimiter(line)
            break
        else:  # pragma: no cover - empty file
            raise ValueError("no numeric data found")

    data = np.loadtxt(path, comments="#", delimiter=delim, usecols=(0, 1))
    if data.ndim != 2 or data.shape[1] < 2:  # pragma: no cover - malformed
        raise ValueError("expected two numeric columns")
    return data[:, 0], data[:, 1]


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
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    return buf.getvalue()

