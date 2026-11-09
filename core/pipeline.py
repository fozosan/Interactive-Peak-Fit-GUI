"""One spectrum's outputs, built and written the same way by the GUI and by batch.

Game plan 2.1 and 2.2. The GUI's Export and batch used to build their own
``_fit.csv`` rows and write their own uncertainty files, and 9 of the 43 fit
columns came out different. Both now hand a fit result
(``core.fit_api.run_fit_consistent``) and the settings it ran with to this
module:

- :func:`peak_records` is the one row builder of ``<base>_fit.csv``;
- :func:`write_spectrum` writes a spectrum's fit table, its trace and its
  uncertainty files, and removes the uncertainty files of an earlier fit that
  this one does not write;
- :func:`finish_uncertainty` turns an uncertainty method's result into what the
  writers and batch's summary take;
- :func:`output_stems` names each batch input's outputs, keeping inputs that
  share a stem apart;
- :func:`unfitted` describes peaks no fit produced, for an export before a fit.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import data_io, models
from .peaks import Peak
from .residuals import build_residual

logger = logging.getLogger(__name__)

# The uncertainty files a spectrum can have, after its output stem. "_band.csv"
# is the name batch gave the bootstrap band before every method's band became
# "_uncertainty_band.csv"; an old one is removed with the rest.
UNCERTAINTY_SUFFIXES = (
    "_uncertainty.csv",
    "_uncertainty_wide.csv",
    "_uncertainty.txt",
    "_uncertainty_band.csv",
    "_band.csv",
)


def file_seed(path, base_seed: Optional[int]) -> int:
    """Seed for one file's fit restarts: the same on every run and machine.

    CRC-32 of the file name, started from the base seed. Python's ``hash()``,
    used before, is salted per process, so the seeds changed on every run.
    """

    return zlib.crc32(Path(path).name.encode("utf-8"), int(base_seed or 0) & 0xFFFFFFFF)


def output_stems(paths: Sequence[str]) -> List[str]:
    """The stem of each input's outputs, in the order of ``paths``.

    An input's own stem, unless another input shares it (``a.csv`` and ``a.txt``,
    compared without case, as Windows does): then ``<stem>_<extension>``, and a
    counter after that if names still clash. Without this the inputs overwrote
    each other's outputs.
    """

    stems = [Path(p).stem for p in paths]
    shared = Counter(s.casefold() for s in stems)
    names = [
        f"{s}_{Path(p).suffix.lstrip('.')}" if shared[s.casefold()] > 1 and Path(p).suffix else s
        for p, s in zip(paths, stems)
    ]
    seen: Counter = Counter()
    out = []
    for name in names:
        seen[name.casefold()] += 1
        out.append(name if seen[name.casefold()] == 1 else f"{name}_{seen[name.casefold()]}")
    return out


def _locked_mask(peaks: Sequence[Any]) -> np.ndarray:
    # No solver fits eta: it is locked with the locked centres and widths.
    return np.array([f for p in peaks for f in (p.lock_center, False, p.lock_width, True)], bool)


def unfitted(x, y, peaks_in: Sequence[Any], baseline, mode: str, mask) -> dict:
    """A fit-shaped record of peaks that no fit produced.

    For an export of peaks placed by hand, from a template, or edited after the
    last fit: ``fit_ok`` is False, no solver set bounds, and the rmse is that of
    these peaks over the window.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    out_peaks = [Peak(p.center, p.height, p.fwhm, p.eta, bool(p.lock_center), bool(p.lock_width)) for p in peaks_in]
    theta = np.array([v for p in out_peaks for v in (p.center, p.height, p.fwhm, p.eta)], float)
    base_fit = None if baseline is None else np.asarray(baseline, float)[mask]
    resid = build_residual(x[mask], y[mask], out_peaks, mode, base_fit, "linear", None)
    r = resid(theta) if theta.size else np.zeros(0)
    locked = _locked_mask(out_peaks)
    n_free = int(np.count_nonzero(~locked))
    return {
        "peaks_out": out_peaks,
        "theta": theta,
        "rmse": float(np.sqrt(np.mean(r * r))) if r.size else float("nan"),
        "fit_ok": False,
        "message": "not fitted",
        "bounds": None,
        "mode": mode,
        "baseline": None if baseline is None else np.asarray(baseline, float),
        "baseline_fit": "fixed",
        "baseline_order": None,
        "joint_baseline": None,
        "noise_model": None,
        "noise_sigma": None,
        "locked_mask": locked,
        "param_names": [f"{n}{i + 1}" for i in range(len(out_peaks)) for n in ("center", "height", "fwhm", "eta")],
        "dof": r.size - n_free if r.size > n_free else float("nan"),
        "restarts_used": float("nan"),
        "jitter_pct": float("nan"),
        "solver_options": {},
    }


def peak_records(file_name: str, fit: Mapping[str, Any], settings: Mapping[str, Any], x) -> List[dict]:
    """The rows of ``<base>_fit.csv`` for one fit, one per peak.

    ``fit`` is ``run_fit_consistent``'s result (or :func:`unfitted`) and
    ``settings`` the settings it ran with. Everything a row says about the fit
    comes from the fit: ``fit_ok``, the solver options it ran with, its
    restarts, and the bounds it held the parameters to (blank when no fit ran).
    """

    fitted = list(fit.get("peaks_out") or [])
    areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in fitted]
    total = sum(areas) or 1.0
    opts = dict(fit.get("solver_options") or {})
    bounds = fit.get("bounds")
    lo, hi = (None, None) if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
    x = np.asarray(x, float)
    base_cfg = dict(settings.get("baseline") or {})
    method = str(base_cfg.get("method", "als")).lower()
    joint = fit.get("baseline_fit") == "joint"
    noise_weighted = bool(fit.get("noise_model")) and fit.get("noise_sigma") is not None
    window = (settings.get("fit_xmin"), settings.get("fit_xmax"))
    fit_lo, fit_hi = (
        (float(window[0]), float(window[1]))
        if None not in window
        else (float(np.min(x)), float(np.max(x)))
    )

    def bound(arr, i, k):
        return float(arr[4 * i + k]) if arr is not None else np.nan

    rows = []
    for i, (p, area) in enumerate(zip(fitted, areas)):
        rows.append(
            {
                "file": file_name,
                "peak": i + 1,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "lock_width": p.lock_width,
                "lock_center": p.lock_center,
                "area": area,
                "area_pct": 100.0 * area / total,
                "rmse": fit.get("rmse", np.nan),
                "fit_ok": bool(fit.get("fit_ok", False)),
                "mode": fit.get("mode"),
                "fit_xmin": fit_lo,
                "fit_xmax": fit_hi,
                "solver_choice": settings.get("solver"),
                "solver_loss": opts.get("loss", np.nan),
                "solver_weight": "noise_model" if noise_weighted else opts.get("weights", np.nan),
                "solver_fscale": opts.get("f_scale", np.nan),
                "solver_maxfev": opts.get("maxfev", np.nan),
                "solver_restarts": fit.get("restarts_used", np.nan),
                "solver_jitter_pct": fit.get("jitter_pct", np.nan),
                "use_baseline": bool(settings.get("use_baseline", True)),
                "baseline_mode": fit.get("mode"),
                "baseline_uses_fit_range": bool(settings.get("baseline_uses_fit_range", True)),
                "perf_numba": bool(settings.get("perf_numba", False)),
                "perf_gpu": bool(settings.get("perf_gpu", False)),
                "perf_cache_baseline": bool(settings.get("perf_cache_baseline", True)),
                "perf_seed_all": bool(settings.get("perf_seed_all", False)),
                "perf_max_workers": int(settings.get("perf_max_workers", 0) or 0),
                "bounds_center_lo": bound(lo, i, 0),
                "bounds_center_hi": bound(hi, i, 0),
                "bounds_fwhm_lo": bound(lo, i, 2),
                "bounds_height_lo": bound(lo, i, 1),
                "bounds_height_hi": bound(hi, i, 1),
                "x_scale": np.nan,
                "baseline_fit": fit.get("baseline_fit", ""),
                "baseline_order": fit.get("baseline_order") if joint else "",
                "baseline_method": method,
                **(
                    {
                        "als_lam": base_cfg.get("lam"),
                        "als_p": base_cfg.get("p"),
                        "als_niter": base_cfg.get("niter"),
                        "als_thresh": base_cfg.get("thresh"),
                    }
                    if method == "als"
                    else dict.fromkeys(("als_lam", "als_p", "als_niter", "als_thresh"), np.nan)
                ),
                **(
                    {
                        "poly_degree": int(base_cfg.get("degree", 2)),
                        "poly_normalize_x": bool(base_cfg.get("normalize_x", True)),
                    }
                    if method == "polynomial"
                    else {"poly_degree": np.nan, "poly_normalize_x": np.nan}
                ),
            }
        )
    return rows


def locks(fit: Mapping[str, Any]) -> List[dict]:
    """Per-peak locks, in the order the uncertainty TXT reports parameters."""

    lm = fit.get("locked_mask")
    if lm is None:
        return []
    lm = np.asarray(lm, bool)
    return [dict(zip(("center", "height", "fwhm", "eta"), map(bool, lm[4 * i : 4 * i + 4]))) for i in range(lm.size // 4)]


def _baseline_record(fit: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    """How the fit treated the baseline, as the uncertainty writers record it."""

    jb = fit.get("joint_baseline")
    if jb is None:
        return {"fit": "fixed", "method": str((settings.get("baseline") or {}).get("method", "als")).lower()}
    out = jb.as_dict()
    out["coef"] = [float(v) for v in np.asarray(fit["baseline_coef"], float)]
    return out


def _pop_baseline_stats(unc_res, n_peak_params: int, names: Sequence[str]) -> dict:
    """Take the joint baseline's coefficients out of a method's statistics.

    The asymptotic method keys them ``p<i>`` (i after the peaks'), the bootstrap
    and the Bayesian by name (``baseline_c<j>``). Returns their estimate, standard
    error and interval, one entry per coefficient, NaN where a method has none.
    """

    stats = unc_res.get("stats") if isinstance(unc_res, dict) else getattr(unc_res, "stats", None)
    out = {"coef": [], "sd": [], "ci_lo": [], "ci_hi": []}
    for j, name in enumerate(names):
        blk = None
        for key in (name, f"p{int(n_peak_params) + j}"):
            got = stats.pop(key, None) if isinstance(stats, dict) else None
            if blk is None and isinstance(got, dict):
                blk = got
        blk = blk or {}

        def _num(*keys):
            for k in keys:
                if blk.get(k) is not None:
                    try:
                        return float(blk[k])
                    except (TypeError, ValueError):
                        continue
            return float("nan")

        out["coef"].append(_num("est"))
        out["sd"].append(_num("sd"))
        out["ci_lo"].append(_num("p2_5", "p2.5", "ci_lo"))
        out["ci_hi"].append(_num("p97_5", "p97.5", "ci_hi"))
    return out


def finish_uncertainty(unc_res: Any, fit: Mapping[str, Any], settings: Mapping[str, Any], method: str) -> dict:
    """An uncertainty method's result, as the writers and batch's summary take it.

    Its label is canonical; the fit's noise model, baseline, rmse and degrees of
    freedom go with it. A joint baseline's coefficients are reported with the
    baseline, never as another peak's parameters. Missing statistics stay NaN,
    which the writers leave blank: a NaN rmse is not 0, and dof <= 0 is not 1.
    """

    baseline_info = _baseline_record(fit, settings)
    if fit.get("joint_baseline") is not None:
        baseline_info.update(
            _pop_baseline_stats(unc_res, int(np.asarray(fit["theta"]).size), fit["joint_baseline"].names())
        )
    unc_norm = data_io.normalize_unc_result(unc_res)
    unc_norm["label"] = data_io.canonical_unc_label(unc_norm.get("label") or method)
    unc_norm["noise_model"] = fit.get("noise_model")
    unc_norm["baseline_model"] = baseline_info
    unc_norm["rmse"] = float(fit.get("rmse", float("nan")))
    # The fit's degrees of freedom (points less the parameters it fitted, eta
    # never among them); the method's own only when the fit has none.
    dof = float("nan")
    for source in (fit.get("dof"), unc_norm.get("dof")):
        try:
            dof = float(source)
        except (TypeError, ValueError):
            continue
        if math.isfinite(dof):
            break
    unc_norm["dof"] = int(dof) if math.isfinite(dof) and dof > 0 else float("nan")
    return unc_norm


def remove_uncertainty_files(base, keep: Iterable[str] = (), log=None) -> None:
    """Delete the uncertainty files under ``base``, which an earlier fit left.

    ``keep`` holds paths never to touch (a batch's inputs), normalised with
    ``os.path.normcase(os.path.abspath(...))``.
    """

    base = Path(base)
    keep = set(keep)
    for suffix in UNCERTAINTY_SUFFIXES:
        old = base.with_name(base.name + suffix)
        if os.path.normcase(os.path.abspath(old)) in keep:
            continue
        try:
            old.unlink(missing_ok=True)
        except OSError as exc:
            msg = f"could not remove stale {old.name}: {exc}"
            logger.warning(msg)
            if log:
                log(msg)


def _band(unc_norm: Mapping[str, Any]):
    band = unc_norm.get("band")
    if band is None:
        band = unc_norm.get("prediction_band")
    if isinstance(band, (tuple, list)) and len(band) == 3:
        return band
    return None


def write_uncertainty(base, file_path, unc_norm: Mapping[str, Any], fit: Mapping[str, Any],
                      settings: Mapping[str, Any]) -> List[Path]:
    """Write ``<base>_uncertainty.csv`` and ``.txt``, the wide CSV when
    ``settings["export_unc_wide"]``, and ``<base>_uncertainty_band.csv`` when the
    method gave a band: one set of files for every method and both paths."""

    base = Path(base)
    long_csv, wide_csv = data_io.write_uncertainty_csvs(
        base, file_path, unc_norm, write_wide=bool(settings.get("export_unc_wide", False))
    )
    written = [Path(long_csv)] + ([Path(wide_csv)] if wide_csv else [])
    txt = base.with_name(base.name + "_uncertainty.txt")
    data_io.write_uncertainty_txt(
        txt,
        unc_norm,
        peaks=fit.get("peaks_out"),
        method_label=unc_norm.get("label", ""),
        file_path=file_path,
        locks=locks(fit),
    )
    written.append(txt)
    band = _band(unc_norm)
    if band is not None:
        band_csv = base.with_name(base.name + "_uncertainty_band.csv")
        level = data_io.ci_level_text(unc_norm.get("alpha", 0.05))
        with band_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["x", f"y_lo{level}", f"y_hi{level}"])
            for xi, lo, hi in zip(*band):
                xi, lo, hi = float(xi), float(lo), float(hi)
                if math.isfinite(xi) and math.isfinite(lo) and math.isfinite(hi):
                    writer.writerow([xi, lo, hi])
        written.append(band_csv)
    return written


def write_spectrum(
    base,
    file_path,
    *,
    x,
    y_raw,
    fit: Mapping[str, Any],
    settings: Mapping[str, Any],
    unc_norm: Optional[Mapping[str, Any]] = None,
    trace: bool = True,
    keep: Iterable[str] = (),
    log=None,
) -> List[Path]:
    """Write one spectrum's files under ``base`` (the path the suffixes follow).

    ``<base>_fit.csv`` from :func:`peak_records`; ``<base>_trace.csv`` when
    ``trace``, with the baseline the fit used; and the uncertainty files of
    ``unc_norm`` (from :func:`finish_uncertainty`). The uncertainty files of an
    earlier fit go first, so the folder never pairs this fit with another
    fit's uncertainty, even when ``unc_norm`` is None. Returns the paths
    written.
    """

    base = Path(base)
    fit_csv = base.with_name(base.name + "_fit.csv")
    with fit_csv.open("w", encoding="utf-8", newline="") as fh:
        fh.write(data_io.build_peak_table(peak_records(Path(file_path).name, fit, settings, x)))
    written = [fit_csv]
    if trace:
        trace_csv = base.with_name(base.name + "_trace.csv")
        with trace_csv.open("w", encoding="utf-8", newline="") as fh:
            fh.write(data_io.build_trace_table(x, y_raw, fit.get("baseline"), fit.get("peaks_out") or []))
        written.append(trace_csv)
    remove_uncertainty_files(base, keep=keep, log=log)
    if unc_norm is not None:
        written.extend(write_uncertainty(base, file_path, unc_norm, fit, settings))
    return written
