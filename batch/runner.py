"""Batch processing orchestrator for Peakfit 3.x.

This module implements a lightweight batch runner that loads spectra matching
glob patterns, applies baseline correction, fits peaks using the selected
solver and writes a combined peak-table CSV. Optionally, per-spectrum trace
tables can also be emitted.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Sequence, List

import numpy as np

from core import data_io, models, peaks, signals
from fit import classic, modern


def _apply_solver(name: str, x, y, pk, mode, baseline, options):
    if name == "classic":
        return classic.solve(x, y, pk, mode, baseline, options)
    if name == "modern":
        return modern.solve(x, y, pk, mode, baseline, options)
    raise ValueError("unknown solver")


def _auto_seed(x: np.ndarray, y: np.ndarray, baseline: np.ndarray, max_peaks: int = 5) -> List[peaks.Peak]:
    """Return up to ``max_peaks`` automatically seeded peaks."""

    sig = y - baseline
    if sig.size < 3:
        return []
    # simple local maxima detection
    mask = (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:])
    idx = np.where(mask)[0] + 1
    if idx.size == 0:
        return []
    prom = sig[idx]
    order = np.argsort(prom)[::-1][:max_peaks]
    idx = idx[order]
    span = float(x.max() - x.min())
    default_w = max(span * 0.05, float(np.mean(np.diff(np.sort(x)))) * 5.0)
    med = float(np.median(sig))
    found: List[peaks.Peak] = []
    for i in idx:
        h = max(float(sig[i] - med), 1e-6)
        found.append(peaks.Peak(float(x[i]), h, float(default_w), 0.5))
    found.sort(key=lambda p: p.center)
    return found


def run(patterns: Iterable[str], config: dict) -> None:
    """Run the peak fitting pipeline over matching files.

    Parameters
    ----------
    patterns:
        Iterable of glob patterns. All matching files are processed.
    config:
        Dictionary describing the batch job. Supported keys include ``peaks``
        (list of peak dictionaries), ``solver`` (``classic`` | ``modern``),
        ``mode`` (``add`` | ``subtract``), ``baseline``
        parameters, per-solver options, ``save_traces`` flag, ``peak_output``
        for the output CSV, ``source`` (``current`` | ``template`` | ``auto``)
        selecting the peak seeds, ``reheight`` to refresh heights per spectrum
        and ``auto_max`` controlling the maximum auto-seeded peaks.
    """

    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise FileNotFoundError("no files matched patterns")

    base_template: Sequence[peaks.Peak] = [
        peaks.Peak(**p) for p in config.get("peaks", [])
    ]
    solver_name = config.get("solver", "classic")
    mode = config.get("mode", "add")
    base_cfg = config.get("baseline", {})
    save_traces = bool(config.get("save_traces", False))
    peak_output = config.get("peak_output", "peaks.csv")
    source = config.get("source", "template")
    reheight = bool(config.get("reheight", False))
    auto_max = int(config.get("auto_max", 5))

    records = []

    for path in files:
        x, y = data_io.load_xy(path)
        baseline = signals.als_baseline(
            y,
            lam=base_cfg.get("lam", 1e5),
            p=base_cfg.get("p", 0.001),
            niter=base_cfg.get("niter", 10),
            tol=base_cfg.get("thresh", 0.0),
        )

        if source == "auto":
            template = _auto_seed(x, y, baseline, max_peaks=auto_max)
        else:
            template = [
                peaks.Peak(p.center, p.height, p.fwhm, p.eta, p.lock_center, p.lock_width)
                for p in base_template
            ]
            if reheight:
                sig = y - baseline
                for tpl in template:
                    idx_near = int(np.argmin(np.abs(x - tpl.center)))
                    tpl.height = float(max(sig[idx_near], 1e-6))

        opts = config.get(solver_name, {})
        res = _apply_solver(solver_name, x, y, template, mode, baseline, opts)

        theta = np.asarray(res["theta"], dtype=float)
        fitted = []
        for i, tpl in enumerate(template):
            c, h, w, e = theta[4 * i : 4 * (i + 1)]
            fitted.append(peaks.Peak(c, h, w, e, tpl.lock_center, tpl.lock_width))

        model = models.pv_sum(x, fitted)
        resid = model + (baseline if mode == "add" else 0.0) - (
            y if mode == "add" else y - baseline
        )
        rmse = float(np.sqrt(np.mean(resid**2)))
        areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in fitted]
        total = sum(areas) or 1.0
        for idx, (p, area) in enumerate(zip(fitted, areas), start=1):
            records.append(
                {
                    "file": Path(path).name,
                    "peak": idx,
                    "center": p.center,
                    "height": p.height,
                    "fwhm": p.fwhm,
                    "eta": p.eta,
                    "lock_width": p.lock_width,
                    "lock_center": p.lock_center,
                    "area": area,
                    "area_pct": 100.0 * area / total,
                    "rmse": rmse,
                    "fit_ok": bool(res["ok"]),
                    "mode": mode,
                    "als_lam": base_cfg.get("lam"),
                    "als_p": base_cfg.get("p"),
                    "fit_xmin": float(x[0]),
                    "fit_xmax": float(x[-1]),
                }
            )

        if save_traces:
            trace_csv = data_io.build_trace_table(
                x, y, baseline, fitted, mode=mode
            )
            trace_path = Path(path).with_suffix(Path(path).suffix + ".trace.csv")
            with trace_path.open("w", encoding="utf-8") as fh:
                fh.write(trace_csv)

    peak_csv = data_io.build_peak_table(records)
    with open(peak_output, "w", encoding="utf-8") as fh:
        fh.write(peak_csv)

