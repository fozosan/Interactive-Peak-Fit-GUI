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

import math
import numpy as np
import pandas as pd

from core import data_io, models, peaks, signals
from core.residuals import build_residual, jacobian_fd
from fit import orchestrator


def _asymptotic_uncertainty(x, y_target, baseline, fitted, mode):
    try:
        theta = []
        for p in fitted:
            theta.extend([p.center, p.height, p.fwhm, p.eta])
        theta = np.asarray(theta, float)
        resid_fn = build_residual(x, y_target, fitted, mode, baseline, "linear", None)
        J = jacobian_fd(resid_fn, theta)
        r0 = resid_fn(theta)
        rss = float(np.dot(r0, r0))
        m = r0.size
        jtj = J.T @ J
        try:
            cov = np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(jtj)
        dof = max(m - theta.size, 1)
        sigma2 = rss / dof
        cov *= sigma2
        sigma = np.sqrt(np.diag(cov))

        def ymodel(th):
            tmp = []
            for i in range(len(fitted)):
                c, h, w, e = th[4 * i : 4 * i + 4]
                tmp.append(peaks.Peak(c, h, w, e))
            total = models.pv_sum(x, tmp)
            if baseline is not None:
                total = total + baseline
            return total

        y0 = ymodel(theta)
        G = np.empty((x.size, theta.size), float)
        for j in range(theta.size):
            step = 1e-6 * max(1.0, abs(theta[j]))
            tp = theta.copy()
            tp[j] += step
            G[:, j] = (ymodel(tp) - y0) / step
        var = np.einsum('ij,jk,ik->i', G, cov, G)
        band_std = np.sqrt(np.maximum(var, 0.0))
        z = 1.96
        lo = y0 - z * band_std
        hi = y0 + z * band_std
        return sigma, (x, lo, hi, y0), {"dof": dof, "s2": sigma2, "rmse": math.sqrt(rss / m)}
    except Exception:
        return np.full(4 * len(fitted), np.nan), None, {"dof": np.nan, "s2": np.nan, "rmse": np.nan}


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


def run(patterns: Iterable[str], config: dict, progress=None, log=None) -> None:
    """Run the peak fitting pipeline over matching files.

    Parameters
    ----------
    patterns:
        Iterable of glob patterns. All matching files are processed.
    config:
        Dictionary describing the batch job. Supported keys include ``peaks``
        (list of peak dictionaries), ``solver`` (``classic`` | ``modern_vp`` |
        ``modern_trf`` | ``lmfit_vp``), ``mode`` (``add`` | ``subtract``), ``baseline``
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
    total = len(files)

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
    unc_rows = []
    ok = 0

    for i, path in enumerate(files, 1):
        if progress:
            progress(i, total, path)
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

        opts = dict(config.get(solver_name, {}))
        opts["solver"] = solver_name
        res = orchestrator.run_fit_with_fallbacks(
            x, y, template, mode, baseline, opts
        )

        theta = np.asarray(res.theta, dtype=float)  # noqa: F841 - for debugging
        fitted = res.peaks_out

        model = models.pv_sum(x, fitted)
        resid = model + (baseline if mode == "add" else 0.0) - (
            y if mode == "add" else y - baseline
        )
        rmse = float(np.sqrt(np.mean(resid**2)))
        areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in fitted]
        total = sum(areas) or 1.0
        perf_extras = {
            "perf_numba": bool(config.get("perf_numba", False)),
            "perf_gpu": bool(config.get("perf_gpu", False)),
            "perf_cache_baseline": bool(config.get("perf_cache_baseline", True)),
            "perf_seed_all": bool(config.get("perf_seed_all", False)),
            "perf_max_workers": int(config.get("perf_max_workers", 0)),
        }
        center_bounds = (
            (float(x[0]), float(x[-1]))
            if (opts.get("centers_in_window") or opts.get("bound_centers_to_window"))
            else (np.nan, np.nan)
        )
        med_dx = float(np.median(np.diff(np.sort(x)))) if x.size > 1 else 0.0
        fwhm_lo = opts.get("min_fwhm", max(1e-6, 2.0 * med_dx))
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
                    "fit_ok": bool(res.success),
                    "mode": mode,
                    "als_lam": base_cfg.get("lam"),
                    "als_p": base_cfg.get("p"),
                    "fit_xmin": float(x[0]),
                    "fit_xmax": float(x[-1]),
                    "solver_choice": solver_name,
                    "solver_loss": opts.get("loss", np.nan),
                    "solver_weight": opts.get("weights", np.nan),
                    "solver_fscale": opts.get("f_scale", np.nan),
                    "solver_maxfev": opts.get("maxfev", np.nan),
                    "solver_restarts": opts.get("restarts", np.nan),
                    "solver_jitter_pct": opts.get("jitter_pct", np.nan),
                    "use_baseline": True,
                    "baseline_mode": mode,
                    "baseline_uses_fit_range": bool(config.get("baseline_uses_fit_range", True)),
                    "als_niter": base_cfg.get("niter"),
                    "als_thresh": base_cfg.get("thresh"),
                    **perf_extras,
                    "bounds_center_lo": center_bounds[0],
                    "bounds_center_hi": center_bounds[1],
                    "bounds_fwhm_lo": fwhm_lo,
                    "bounds_height_lo": 0.0,
                    "bounds_height_hi": np.nan,
                    "x_scale": opts.get("x_scale", np.nan),
                }
            )

        if res.success:
            y_target = y if mode == "add" else y - baseline
            base_resid = baseline if mode == "add" else None
            sigma, _band, info = _asymptotic_uncertainty(x, y_target, base_resid, fitted, mode)
        else:
            sigma = np.full(4 * len(fitted), np.nan)
            info = {"dof": np.nan, "rmse": rmse}
        z = 1.96
        for idx, p in enumerate(fitted, start=1):
            sc = sigma[4 * (idx - 1)] if sigma.size >= 4 * idx else np.nan
            sh = sigma[4 * (idx - 1) + 1] if sigma.size >= 4 * idx + 1 else np.nan
            sf = sigma[4 * (idx - 1) + 2] if sigma.size >= 4 * idx + 2 else np.nan
            if p.lock_center:
                sc = np.nan
            if p.lock_width:
                sf = np.nan
            params = [
                ("center", p.center, sc, not p.lock_center),
                ("height", p.height, sh, True),
                ("fwhm", p.fwhm, sf, not p.lock_width),
                ("eta", p.eta, np.nan, False),
            ]
            for pname, val, std, free in params:
                if free and np.isfinite(std):
                    ci_lo, ci_hi = val - z * std, val + z * std
                else:
                    ci_lo = ci_hi = np.nan
                unc_rows.append(
                    {
                        "file": Path(path).name,
                        "peak": idx,
                        "param": pname,
                        "value": val,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "method": "asymptotic",
                        "rmse": rmse,
                        "dof": info.get("dof", np.nan),
                    }
                )

        trace_path = None
        if save_traces:
            trace_csv = data_io.build_trace_table(
                x, y, baseline, fitted
            )
            trace_path = Path(path).with_suffix(Path(path).suffix + ".trace.csv")
            with trace_path.open("w", encoding="utf-8") as fh:
                fh.write(trace_csv)
        if log:
            msg = f"{Path(path).name}: {'ok' if res.success else 'fail'} rmse={rmse:.3g}"
            if trace_path:
                msg += f" {trace_path}"
            log(msg)
        if res.success:
            ok += 1

    peak_csv = data_io.build_peak_table(records)
    with open(peak_output, "w", encoding="utf-8") as fh:
        fh.write(peak_csv)
    unc_output = config.get("unc_output")
    if not unc_output:
        unc_output = Path(peak_output).with_name(Path(peak_output).stem + "_uncertainty.csv")
    pd.DataFrame(unc_rows).to_csv(unc_output, index=False)

    return ok, total
