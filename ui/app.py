#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive peak fit GUI for spectra (Gaussian–Lorentzian / pseudo-Voigt)
Designed by Farhan Zahin • Built with ChatGPT

Build: v3.2-beta

What’s new since v3.0 / v3.1
• Reliability & parity: Classic is restored as a simple SciPy curve_fit backend (lock-aware, minimal bounds). Modern TRF/VP stabilized (robust losses/weights; VP heights via NNLS).
• “Step ▶” parity: Step uses the same residual/bounds as Fit and reports damping λ, backtracks, step_norm, and accept/reject reasons. Non-finite guards prevent freezes.
• Persisted settings: Shape factor η, “Add peaks on click”, solver choice, uncertainty method, performance toggles, legend visibility, and defaults (e.g., labels) are saved to ui/ui_config/gl_peakfit_config.json, beside the code, so OneDrive syncs them (the old ~/.gl_peakfit_config.json is read until that file exists, and never changed).
• UI/UX: Segmented action bar (File | Fit | Plot | Help) is always visible; F1 opens Help. Right panel scrolling is independent from the Help window; splitter enforces sensible min widths; status bar with progress + collapsible green-on-black log; batch streams progress.
• Defaults: “Show uncertainty band” = ON. “Baseline uses fit range” = ON. Performance toggles persist (Numba/cache/seed/uncertainty workers).
• Legend: Toggleable legend with peak centers in labels; legend font set to Arial.
• Export parity: Canonical per-file outputs with no blank lines — *_fit.csv, *_trace.csv, *_uncertainty.csv, *_uncertainty.txt — identical schemas for single and batch; batch writes to the chosen output folder plus a batch summary.
• Uncertainty: Asymptotic band via SVD covariance (smoothed prediction band), Bootstrap (residual resampling with seeded determinism and worker cap), and optional Bayesian (emcee) with ESS/R-hat diagnostics and smoothed bands.
• Batch parity fixes: deep-copy seeds, enforce locks/bounds, deterministic jitter respecting locks, per-file baseline computed inside the fit window; single–batch RMSE/θ parity verified on fixtures.

Overview (user workflow)
========================
• Load 2-column spectra (x, y) from CSV/TXT/DAT. Delimiters auto-detected; lines starting with #, %, // and text headers are ignored. Non-finite rows dropped; x is sorted if needed.
• Baseline (ALS): choose λ (smoothness), p (asymmetry), max iterations, and an early-stop threshold. By default ALS is computed ONLY on the selected fit window and interpolated across full x.
• Fit modes:
  – Add: model = baseline + Σ peaks, fitted directly to raw y (WYSIWYG).
  – Subtract: model = Σ peaks, fitted to (y − baseline).
• Peaks: click to add (height auto-initialized from local signal), or auto-seed prominent peaks in the fit window. Lock width and/or center per peak. η is per-peak but can be broadcast to all.
• Solvers:
  – Classic (curve_fit): simple unweighted LSQ with minimal bounds; respects locks.
  – Modern TRF: robust least_squares (loss = linear/soft_l1/huber/cauchy), optional per-point noise weights (none/poisson/inv_y).
  – Modern VP: variable-projection; heights by nonnegative LS (NNLS), centers/widths by Gauss–Newton with backtracking.
  – LMFIT-VP (optional): if lmfit is installed, exposes a comparable VP pipeline.
• Step ▶: one iteration of the currently selected solver; same residuals/weights/bounds; acceptance with backtracking & λ diagnostics.
• Batch: process folders with patterns; seed from current/template/auto (templates auto-apply); optional re-height per file; optional per-spectrum traces; unified summary CSV.
• Exports: unified peak table (single & batch share schema) + trace CSVs (both “added” and “subtracted” sections) + uncertainty CSV/TXT with ±.

Mathematical model
==================
Pseudo-Voigt peak:
  g(x; h, c, w, η) = h * [(1−η) * exp(−4 ln 2 * ((x−c)/w)^2)  +  η * 1/(1 + 4((x−c)/w)^2)]
where h≥0 is height, c is center, w>0 is FWHM, and η∈[0,1] is the GL mix factor (0=Gaussian, 1=Lorentzian).

Model & residuals
=================
• Let P(x; θ) = Σ_j g_j(x; h_j, c_j, w_j, η_j).
• Add mode:    y_fit = baseline + P,     r = (y_fit − y_raw).
• Subtract:    y_fit = P,                r = (y_fit − (y_raw − baseline)).
• Locks: if a peak has lock_center/lock_width, that parameter is held constant and not optimized.
• Bounds: heights ≥ 0; widths ≥ eps_w (data-driven or 1e−6); centers optionally clamped to the fit window for Classic. Centers/widths for Modern use solver bounds directly.

Noise / robust weighting
========================
• Noise weights (optional): w_noise ∈ {none, poisson, inv_y}.
  – poisson: w_i = 1 / √max(|y_target_i|, ε)   (ε avoids division by zero)
  – inv_y:   w_i = 1 / max(|y_target_i|, ε)
• Robust losses (Modern/TRF): SciPy least_squares with loss ∈ {linear, soft_l1, huber, cauchy} and f_scale.
  – Residuals are internally scaled by f_scale per SciPy’s robust formulation; J and the solver handle the M-estimator.
• Combined weights: residuals and Jacobians are premultiplied by diag(w_noise) before calling the solver. IRLS passes (when enabled) recompute weights after trial steps/backtracks.

Solvers (implementation details)
================================
Classic (SciPy curve_fit)
• Pack only free parameters (heights always free; c/w free unless locked).
• Build the unweighted residual r(θ) consistent with the chosen mode (Add/Subtract) and baseline slice on the fit window.
• Bounds: h ≥ 0, w ≥ eps_w; centers optionally restricted to the window. curve_fit handles Jacobians by finite differences.
• On success, unpack θ back to peaks. The Step ▶ path mirrors the same residual/bounds and performs a single damped Gauss–Newton step with backtracking (accept on cost↓).

Modern TRF (SciPy least_squares)
• The same residual r(θ) but weighted and with robust loss. Parameters are scaled (x_scale) for conditioning.
• Jacobian: finite differences by default (good stability with TRF). Optional analytic blocks can be injected when available.
• Backtracking & damping handled by least_squares; Step ▶ uses a single internal iteration with the same objective.

Modern VP (variable-projection)
• Split parameters into linear (heights) and nonlinear (centers/widths). Given current (c,w), form a design matrix A with columns A[:,j] = g_j(x; h=1, c_j, w_j, η_j).
• Solve heights by **NNLS** on the weighted system:  h = argmin_{h≥0} || W(A h − y_target) ||_2.
• With heights fixed, update (c,w) via a Gauss–Newton step on the reduced objective; step-halving backtracking; bounds/locks respected.
• Jacobians: columns for ∂P/∂c_j and ∂P/∂w_j are approximated via finite differences of g_j with unit height (stable and simple); analytics can be plugged in later.
• Step ▶ calls the same VP iterate once, with the same weighting and backtracking rules.

LMFIT-VP (optional)
• If lmfit is installed, a thin wrapper builds Parameters for the same θ packing and uses its least_squares engine. Heights may be solved by LS per iteration or via NNLS when enabled.

ALS baseline (Eilers & Boelens)
================================
We solve z = argmin ||W(y − z)||^2 + λ ||D^2 z||^2 with asymmetric weights W from p (pushes baseline under peaks). Sparse tri-diagonal operator D^2 is used; we iterate:
1) Solve (W + λ DᵀD + εI) z = W y
2) Update W_i = p if y_i > z_i else (1−p)
3) Stop when max|Δz| / (max|z_old|+1e−12) ≤ threshold or when max iterations reached.
If “baseline uses fit window” is enabled, we compute z only on the window and interpolate over full x (constant extrapolation at ends).

Step ▶ engine (shared contract)
===============================
Given state (θ, residual function r(θ), bounds, weights), perform one iteration:
1) Compute J and r at θ (finite differences unless analytic is available).
2) Try a damped Gauss–Newton / LM step: solve (JᵀJ + λI) δ = −Jᵀr for δ (stable solve with lstsq fallback).
3) Backtracking: evaluate cost at θ+δ; if not decreased, increase λ (and/or halve the step) and retry up to max_backtracks.
4) Accept if cost↓ and δ not tiny; return diagnostics: accepted flag, λ_used, backtracks, step_norm, and reason (“ok”, “tiny_step”, “nan_guard”, “max_backtracks”).
The engine shares residuals/weights/bounds with the full Fit so Step ▶ visually “walks” the same path.

Uncertainty (asymptotic, bootstrap, Bayesian)
=============================================
Asymptotic (default band ON):
• At the current solution θ*, compute J on the fit window and residual vector r (with the chosen weights).
• Estimate σ² = RSS / dof, where RSS = ||r||² and dof = max(1, m − n); Cov ≈ σ² (JᵀJ)⁻¹ with tiny Tikhonov if needed.
• Delta method for the predicted curve ŷ(x): form G = ∂ŷ/∂θ (FD on full x). Var[Ŷ] = diag(G Cov Gᵀ); 95% CI = ŷ ± 1.96 √Var (band is lightly smoothed for display).

Bootstrap:
• Residual bootstrap with refits: y* = y_fit + rⱼ* (resampled residuals) on the fit window; refit to obtain θ* draws and optional ŷ* bands.
• Reproducible with fixed seeds; uses the configured worker cap; respects locks/bounds and baseline-in-window behavior.

Bayesian (optional; emcee):
• MCMC over free parameters; reports posterior mean/SD and 95% credible intervals, plus convergence diagnostics (ESS/R-hat).
• Prediction band from posterior predictive; band lightly smoothed. If emcee is absent a “NotAvailable” status is returned.

Locks, bounds, and parameter packing
====================================
• Each peak has (h, c, w, η). η is user-set (not varied by solvers) but may be broadcast. Locked parameters are not included in θ and are kept fixed.
• Bounds: Classic uses minimal bounds (h≥0, w≥eps_w, centers optional window clamp). Modern solvers supply scipy bounds directly; heights are constrained non-negative in VP via NNLS.
• All solvers unpack θ back to peaks in a lock-aware order; Step ▶ uses the same pack/unpack.

File I/O
========
Import
• Robust reader tries pandas with sep=None (auto detect), coerces numerics, drops non-numeric/text columns, and filters non-finite rows. If pandas fails, a manual parser strips comments (#, %, //) and whitespace/semicolon delimiters.
• x is sorted ascending if needed. A two-column numeric dataset is required.

Export (single & batch share formats; no blank lines)
• Peak table CSV columns (fixed order):
  file, peak, center, height, fwhm, eta, lock_width, lock_center,
  area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax,
  solver_choice, solver_loss, solver_weight, solver_fscale, solver_maxfev,
  solver_restarts, solver_jitter_pct, step_lambda,
  baseline_uses_fit_range, perf_numba, perf_gpu, perf_cache_baseline,
  perf_seed_all, perf_max_workers
  – area: closed-form pseudo-Voigt area combining Gaussian and Lorentzian parts
  – area_pct: 100 × area / Σ area
  – rmse: computed on the active fit window against the proper target (Add: raw; Subtract: raw − baseline)
  – mode: "add" or "subtract"; als_* are the baseline parameters used; fit_x* are the window limits (or empty if full range)

• Trace CSV columns (fixed order):
  x, y_raw, baseline,
  y_target_add, y_fit_add, peak1, peak2, …,
  y_target_sub, y_fit_sub, peak1_sub, peak2_sub, …
  – peakN      = baseline-ADDED component (for plotting like “Add” mode)
  – peakN_sub  = baseline-SUBTRACTED pure component (for calculations)
  The “added” block appears first to match the display; the “subtracted” block follows.

• Uncertainty outputs:
  – *_uncertainty.csv: tabular parameter stats (mean, sd, CI) and, when requested, band summaries.
  – *_uncertainty.txt: human-readable report with ± values and notes about the method/diagnostics.

Persistence & defaults
======================
The configuration file (ui/ui_config/gl_peakfit_config.json, beside the code: git-ignored and synced by OneDrive; the GL_PEAKFIT_CONFIG environment variable names another file) stores: ALS defaults (λ, p, iterations, threshold), solver choice & options (loss/weights/f_scale/maxfev/restarts/jitter/step_lambda), uncertainty method, click-to-add toggle, global η, performance toggles (numba/cache/seed_all/uncertainty workers), batch defaults, x-label, legend visibility, and templates (including auto-apply). Settings update on change and are loaded at startup. Until that file exists, the old ~/.gl_peakfit_config.json is loaded in its place; the first save writes the new file, and the old one is never changed.

UI details
==========
• Action bar: File [Open, Export, Batch] | Fit [Step, Fit] | Plot [Uncertainty, Legend, Components] | Help (F1).
• Legend font is Arial. Log uses green text on black. The right-hand control panel is fully scrollable.

Performance notes
=================
• Conditioning: TRF uses x_scale to normalize parameter sensitivities; Classic keeps minimal bounds for speed.
• VP reduces ill-conditioning by solving linear heights separately via NNLS at each nonlinear update.
• Baseline solves use sparse operators; IRLS converges quickly for reasonable λ/p. Early-stop threshold avoids unnecessary passes.
• Optional Numba/CuPy accelerations and caching are available with safe fallbacks; shadow-compare is used in tests for safety.

Testing & reproducibility
=========================
• Regression tests enforce that Modern TRF/VP and (optionally) LMFIT-VP reproduce reference costs/θ on fixtures; Classic has dedicated lock/bounds and step-vs-solve parity tests.
• Single–batch parity tests confirm equivalent RMSE/parameters; exports are checked to have no blank lines and stable schemas.
• For restarts/jitter paths, fix RNG seeds for reproducibility; batch honors per-file baseline windows and seed handling.

Known limitations / tips
========================
• Classic is intentionally simple (no robust loss/weights). Use Modern TRF for outliers/heavy tails or Modern VP for overlapped peaks.
• If Step ▶ is repeatedly rejected, try a smaller λ or a better start; for VP, ensure heights remain non-negative and widths above eps_w.
• If ALS rides peak tops, increase λ and/or decrease p. Using the fit window for ALS often improves local baselines.
• If uncertainty bands show spikes, narrow the fit window, lock weakly determined parameters, try robust losses, or increase bootstrap samples; Bayesian requires emcee and adequate sampling.
"""

import json
import logging
import math  # used for band row filtering
import queue
import re
import shutil
import tempfile
import time
import csv
from dataclasses import dataclass
from functools import wraps  # used by @log_action
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from types import SimpleNamespace  # ensure this import exists
try:
    import core.data_io as _dio  # for build_peak_table/trace_table in batch exports
except Exception:
    # Will raise at first use if missing; kept non-fatal to avoid breaking app start
    _dio = None  # type: ignore

import numpy as np

import os
import matplotlib
if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
matplotlib.rcdefaults()
matplotlib.rcParams["font.family"] = "Arial"
# Use native fonts with better rendering on modern displays (no TeX dependency)
matplotlib.rcParams["text.usetex"] = False
# Higher DPI for export quality on modern displays
matplotlib.rcParams["savefig.dpi"] = 150

# Safe antialiasing rcParams guards - only set if keys exist in current matplotlib version
for _key in ("text.antialiased", "lines.antialiased", "patch.antialiased"):
    if _key in matplotlib.rcParams:
        matplotlib.rcParams[_key] = True

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

# ---- Threadpool runtime control (optional) ----
try:
    from threadpoolctl import threadpool_limits, threadpool_info
except Exception:
    threadpool_limits = None

    def threadpool_info():  # type: ignore
        return []

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import simpledialog
import tkinter.font as tkfont
import threading
import traceback

from scipy.signal import find_peaks

from core import signals
from ui.theme import apply_theme, log_theme_status, paint_icon
# --- Canonical uncertainty label: delegate to data_io when available ---
try:
    _canonical_unc_label = _dio.canonical_unc_label  # type: ignore[attr-defined]
except Exception:
    _canonical_unc_label = lambda s: str(s).strip().lower()
from ui.uncertainty_utils import (
    method_display_to_key,
    normalize_float,
    normalize_int,
    validate_unc_knobs,
)
try:
    from core.uncertainty import UncertaintyResult, NotAvailable
    from core.uncertainty import (
        BAND_SKIP_DIAG_DISABLED,
        BAND_SKIP_DIAG_UNHEALTHY,
        BAND_SKIP_INSUFF_DRAWS,
        BAND_SKIP_ABORTED,
    )
except Exception:  # pragma: no cover - NotAvailable may be absent
    from core.uncertainty import UncertaintyResult  # type: ignore

    class NotAvailable:  # pragma: no cover - fallback placeholder
        pass
from core import fit_api, pipeline
from core import peaks as core_peaks
from core.fit_api import (
    classic_step,
    modern_trf_step,
    modern_vp_step,
    lmfit_step,
)
from fit import classic, modern_vp, modern
try:  # optional
    from fit import lmfit_backend
except Exception:  # pragma: no cover - optional dependency may be missing
    lmfit_backend = None


# ---------- Debug helpers (stateful, low-noise) ----------
def _kv(d: dict) -> str:
    try:
        return " ".join(f"{k}={v}" for k, v in d.items())
    except Exception:
        return str(d)


def _safe_bool(v, default=False):
    try:
        return bool(v)
    except Exception:
        return default


def _finite_or_nan(x):
    try:
        return float(x) if np.isfinite(float(x)) else float("nan")
    except Exception:
        return float("nan")


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _set_thread_limits(strategy: str, blas_threads):
    """UI helper: actual clamping is handled in the performance layer."""

    return


def log_action(fn):
    """Decorator to emit debug log lines when debug mode is on."""

    @wraps(fn)
    def _wrap(self, *a, **k):
        try:
            if _safe_bool(self.debug_mode_var.get(), False):
                def _srepr(x, n=120):
                    try:
                        s = repr(x)
                    except Exception:
                        s = object.__repr__(x)
                    return s if len(s) <= n else s[:n] + "\u2026"

                args_s = ", ".join(_srepr(v) for v in a)
                kwargs_s = ", ".join(f"{kk}={_srepr(vv)}" for kk, vv in k.items())
                self.log_threadsafe(
                    f"[DEBUG] {fn.__name__}({args_s}" + (", " if kwargs_s else "") + f"{kwargs_s})"
                )
        except Exception:
            pass
        return fn(self, *a, **k)

    return _wrap

BACKENDS = {
    "classic": classic,
    "modern_vp": modern_vp,
    "modern_trf": modern,
}
if lmfit_backend is not None:
    BACKENDS["lmfit_vp"] = lmfit_backend
from infra import performance
from batch import runner as batch_runner

MODERN_LOSSES = ["linear", "soft_l1", "huber", "cauchy"]
MODERN_WEIGHTS = ["none", "poisson", "inv_y"]
LMFIT_ALGOS = ["least_squares", "leastsq", "nelder", "differential_evolution"]

SOLVER_LABELS = {
    "classic": "Classic (curve_fit)",
    "modern_vp": "Modern (Variable Projection)",
    "modern_trf": "Modern (Legacy TRF)",
    "lmfit_vp": "LMFIT (Variable Projection)",
}

STEP_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAATElEQVR4nM3TOQ4AIAwDQYL4/5ehoskh26HBVYS00zHGT9udaCaIBHlAhiqAhhAAIRYoIRUI0GoCdg8V"
    "MP/AAiFkgTJEAAyztf7C8w5Q0w4YsEzvsQAAAABJRU5ErkJggg=="
)
SOLVER_LABELS_INV = {v: k for k, v in SOLVER_LABELS.items()}


# ---------- Math ----------
def gaussian(x, x0, fwhm):
    return np.exp(-4.0 * np.log(2.0) * ((x - x0) ** 2) / (fwhm ** 2))

def lorentzian(x, x0, fwhm):
    return 1.0 / (1.0 + 4.0 * ((x - x0) ** 2) / (fwhm ** 2))

def pseudo_voigt(x, height, x0, fwhm, eta):
    eta = np.clip(eta, 0.0, 1.0)
    return height * ((1.0 - eta) * gaussian(x, x0, fwhm) + eta * lorentzian(x, x0, fwhm))

def pseudo_voigt_area(height, fwhm, eta):
    ga_area = height * fwhm * math.sqrt(math.pi / (4 * math.log(2.0)))
    lo_area = height * (math.pi * fwhm / 2.0)
    eta = np.clip(eta, 0.0, 1.0)
    return (1.0 - eta) * ga_area + eta * lo_area


# ---------- Data structures ----------
@dataclass
class Peak:
    center: float
    height: float = 1.0
    fwhm: float = 5.0
    eta: float = 0.5
    lock_width: bool = False
    lock_center: bool = False
    # Optional per-peak cap on FWHM when width is *not* locked
    fwhm_cap_enabled: bool = False
    fwhm_cap_value: float = 50.0


# ---------- Config persistence ----------
def _config_paths():
    """Return (settings file, old file to read while the settings file is absent).

    Settings live beside the code, in ui/ui_config/, because OneDrive syncs the repo
    folder between machines and $HOME is neither synced nor backed up. The file is
    git-ignored: personal, not published. GL_PEAKFIT_CONFIG names another file, for
    tests and tools, and has no old file. Settings used to live in
    ~/.gl_peakfit_config.json: load_config reads it until the new file exists, and
    nothing writes to it. gl_decon_gui_batch.py uses the same file.
    """
    override = os.environ.get("GL_PEAKFIT_CONFIG")
    if override:
        return Path(override), None
    return (
        Path(__file__).resolve().parent / "ui_config" / "gl_peakfit_config.json",
        Path.home() / ".gl_peakfit_config.json",
    )


CONFIG_PATH, LEGACY_CONFIG_PATH = _config_paths()
# The app wrote its defaults into every settings file at startup, so a stored value
# is no sign of a choice. When a default changes, files from before carry the old
# one; _upgrade_settings resets those values once, and the file then carries the
# version, after which what the user sets is kept.
SETTINGS_VERSION = 2
DEFAULTS = {
    "settings_version": SETTINGS_VERSION,
    "baseline_defaults": {
        "method": "als",
        "als": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "polynomial": {"degree": 2, "normalize_x": True},
    },
    # Legacy single template (migrated to templates/default if present)
    "saved_peaks": [],
    # Multiple templates live here as {"name": [peak dicts...]}
    "templates": {},
    "auto_apply_template": False,
    "auto_apply_template_name": "",
    "x_label": "x",
    "batch_patterns": "*.csv;*.txt;*.dat",
    "batch_source": "template",
    "batch_reheight": False,
    "batch_auto_max": 5,
    "batch_save_traces": False,
    # Default solver backend
    "solver_choice": "modern_vp",
    "ui_eta": 0.5,
    # Persisted UI prefs for FWHM caps
    "ui_fwhm_cap_all": False,
    "ui_fwhm_cap_value": 50.0,
    "ui_add_peaks_on_click": True,
    "unc_method": "asymptotic",
    "uncertainty": {"method": "asymptotic", "return_band": True},
    "x_label_auto_math": True,
    "ui_show_legend": True,
    "legend_center_sigfigs": 6,
    # Uncertainty and performance defaults
    "show_uncertainty_band": True,
    "baseline_uses_fit_range": True,
    "perf_numba": False,
    "perf_cache_baseline": True,
    "perf_seed_all": False,
    # Font configuration defaults (added for font rendering improvements)
    "ui_font_family": "auto",
    "ui_font_size": 10,
    "ui_mono_font_family": "auto",
    "font_diagnostics": False,
}

LOG_MAX_LINES = 5000
_config_log = logging.getLogger(__name__)
# What the last load_config changed in an older file; the app logs these once.
_SETTINGS_NOTES: List[str] = []


def _upgrade_settings(cfg: dict, version: int) -> None:
    """Bring the settings of a file written by an older version up to this one.

    Version 2: the bootstrap start jitter's default became 0 (PR #248). It scales
    with each parameter's value, so at 2 % a centre at 383 cm-1 started 7.7 cm-1
    away, the refits landed in other minima and the intervals came out 4 to 4500x
    too wide. An older file holds the old default, which the app wrote into every
    file, or a value chosen on that scale before this was measured: it starts at
    0, once, and the app says so. The Bayesian default became 8000 steps (PR
    #252), so the old default, 4000, becomes 8000; another stored value is kept.
    """

    if version < 2:
        old = cfg.get("bootstrap_jitter")
        cfg["bootstrap_jitter"] = 0.0
        try:
            old_pct = 100.0 * float(old)
        except (TypeError, ValueError):
            old_pct = 0.0
        if old_pct:
            note = (
                f"Settings: the bootstrap start jitter is now 0, the default (it was {old_pct:g} %). "
                "It scales with each parameter's value, so the refits started far from the fit and "
                "landed in other minima: at 2 % the intervals came out 4 to 4500x too wide. "
                "To use one anyway, set the bootstrap Jitter % under Uncertainty again."
            )
            _config_log.warning(note)
            _SETTINGS_NOTES.append(note)
        if cfg.get("bayes_steps") == 4000:
            cfg["bayes_steps"] = 8000


def _read_settings(path):
    """The settings in ``path`` as a dict; raises when the file does not hold a JSON object."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, found {type(data).__name__}")
    return data


def _set_aside_unreadable(path, err, then):
    """Rename a settings file that cannot be read, so that no save writes over it, and warn.

    The new name keeps the suffix, so the ui/ui_config/*.json git-ignore rule still covers
    it. Returns True when ``path`` is free to write.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    aside, n = path.with_name(f"{path.stem}.unreadable-{stamp}{path.suffix}"), 1
    while aside.exists():
        n += 1
        aside = path.with_name(f"{path.stem}.unreadable-{stamp}-{n}{path.suffix}")
    try:
        os.rename(path, aside)
        moved = True
        msg = f"The settings file {path} could not be read ({err}). It was kept as {aside.name}, and {then}."
    except OSError as move_err:
        moved = False
        msg = (f"The settings file {path} could not be read ({err}) or moved aside ({move_err}). "
               "It is left as it is, and settings are not saved over it.")
    _config_log.warning(msg)
    messagebox.showwarning("Settings", msg)
    return moved


def _replace_settings(src, dst, attempts=5):
    # Windows refuses to replace a file that another program, such as a sync client or a
    # virus scanner, holds open for a moment.
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.05 * (k + 1))


def load_config():
    _SETTINGS_NOTES.clear()
    path = CONFIG_PATH
    if not path.exists() and LEGACY_CONFIG_PATH is not None and LEGACY_CONFIG_PATH.exists():
        path = LEGACY_CONFIG_PATH  # read once; the next save_config writes CONFIG_PATH
    if path.exists():
        try:
            data = _read_settings(path)
            cfg = {**DEFAULTS, **data}
            # Ensure nested uncertainty config exists and is independent of defaults
            unc_section = dict(cfg.get("uncertainty") or {})
            legacy_method = data.get("unc_method", cfg.get("unc_method"))
            if "method" not in unc_section:
                method = str(legacy_method or unc_section.get("method") or "asymptotic").strip().lower()
                if method.startswith("boot"):
                    method = "bootstrap"
                elif method.startswith("bayes"):
                    method = "bayesian"
                elif not method:
                    method = "asymptotic"
                unc_section["method"] = method
            legacy_band = data.get("ui_show_uncertainty_band")
            if legacy_band is None:
                legacy_band = data.get("show_uncertainty_band")
            if "return_band" not in unc_section:
                unc_section["return_band"] = bool(legacy_band) if legacy_band is not None else bool(unc_section.get("return_band", True))
            cfg["uncertainty"] = unc_section
            # Maintain legacy alias for compatibility
            cfg["unc_method"] = unc_section.get("method", "asymptotic")
            # Migration: move legacy saved_peaks into templates["default"] if templates is empty
            if cfg.get("saved_peaks") and not cfg.get("templates"):
                cfg["templates"] = {"default": cfg["saved_peaks"]}
            # Migration: legacy solver names
            sc = cfg.get("solver_choice")
            if sc == "modern":
                cfg["solver_choice"] = "modern_vp"
            elif sc == "lmfit":
                cfg["solver_choice"] = "lmfit_vp"
            try:
                version = int(data.get("settings_version", 1))
            except (TypeError, ValueError):
                version = 1
            _upgrade_settings(cfg, version)
            cfg["settings_version"] = SETTINGS_VERSION
            cfg.pop("ui_theme", None)
            # Migration: if the user file lacked baseline_defaults, construct
            # one from any legacy ALS keys present.  We look at the original
            # ``data`` rather than the merged cfg so that default values don't
            # mask legacy settings the user supplied.
            if "baseline_defaults" not in data:
                has_legacy = any(
                    k in data for k in ("als_lam", "als_asym", "als_niter", "als_thresh")
                )
                if has_legacy:
                    als_block = {
                        "lam": float(cfg.get("als_lam", 1e5)),
                        "p": float(cfg.get("als_asym", 0.001)),
                        "niter": int(cfg.get("als_niter", 10)),
                        "thresh": float(cfg.get("als_thresh", 0.0)),
                    }
                    existing_bd = cfg.get("baseline_defaults", {})
                    poly_block = (
                        existing_bd.get("polynomial") if isinstance(existing_bd, dict) else None
                    ) or {"degree": 2, "normalize_x": True}
                    cfg["baseline_defaults"] = {
                        "method": "als",
                        "als": als_block,
                        "polynomial": poly_block,
                    }
            return cfg
        except Exception as err:
            if path == CONFIG_PATH:  # the start saves at once, and would write over it
                _set_aside_unreadable(path, err, "the app starts from the default settings")
    cfg = dict(DEFAULTS)
    cfg["uncertainty"] = dict(cfg.get("uncertainty") or {})
    return cfg

def save_config(cfg):
    """Write the settings so that a save that fails part-way leaves the previous file whole.

    The JSON goes to a temporary file in the same folder and is flushed to disk before
    os.replace swaps it in. A file already there that cannot be read is set aside first,
    never written over.
    """
    path, tmp = CONFIG_PATH, None
    try:
        text = json.dumps(cfg, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                _read_settings(path)
            except Exception as err:
                if not _set_aside_unreadable(path, err, "the current settings are saved in its place"):
                    return
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=f".saving{path.suffix}", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            shutil.copymode(path, tmp)
        _replace_settings(tmp, path)
        tmp = None
    except Exception as e:
        messagebox.showwarning("Config", f"Could not save config: {e}")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def add_tooltip(widget, text: str) -> None:
    """Attach a simple tooltip to ``widget`` displaying ``text``."""

    def on_enter(_e):
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 10
        tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tip, text=text, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        lbl.pack()
        widget._tip = tip

    def on_leave(_e):
        tip = getattr(widget, "_tip", None)
        if tip is not None:
            tip.destroy()
            widget._tip = None

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


# ---------- Label formatting ----------
def format_axis_label_inline(text: str, enabled: bool = True) -> str:
    r"""Convert ``^``/``_`` fragments to inline math while preserving normal text.

    If ``enabled`` is ``False`` or the entire string is already a single
    ``$...$`` math block, the text is returned unchanged.  Existing
    ``$...$`` regions are kept as-is, and escaped literals ``\^`` and ``\_``
    remain literal.
    """

    if not enabled:
        return text
    if re.fullmatch(r"\$[^$]*\$", text.strip()):
        return text

    parts = re.split(r"(\$[^$]*\$)", text)
    out = []

    ESC_CARET = "\0CAR\0"
    ESC_UND = "\0UND\0"

    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
            continue

        tmp = part.replace(r"\^", ESC_CARET).replace(r"\_", ESC_UND)

        tmp = re.sub(r"(?<!\$)\^\s*\{([^{}]+)\}", lambda m: "$^{" + m.group(1).strip() + "}$", tmp)
        tmp = re.sub(r"(?<!\$)\^\s*([+\-]?\d+(?:\.\d+)?)", lambda m: "$^{" + m.group(1) + "}$", tmp)
        tmp = re.sub(r"(?<!\$)\^\s*(\w+)", lambda m: "$^{" + m.group(1) + "}$", tmp)
        tmp = re.sub(
            r"(?<!\$)_\s*\{([^{}]+)\}",
            lambda m: ("$_" + val + "$") if (val := m.group(1).strip()).isalnum() else ("$_{" + val + "}$"),
            tmp,
        )
        tmp = re.sub(
            r"(?<!\$)_(\w+)",
            lambda m: ("$_" + m.group(1) + "$") if len(m.group(1)) == 1 else ("$_{" + m.group(1) + "}$"),
            tmp,
        )

        tmp = tmp.replace(ESC_CARET, "^").replace(ESC_UND, "_")
        out.append(tmp)

    return "".join(out)

# ---------- Scrollable frame ----------
class ScrollableFrame(ttk.Frame):
    """A ttk.Frame that contains a vertically scrollable interior frame."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        self.interior.bind("<Configure>", self._on_interior_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Local wheel state
        self._wheel_accum = 0
        self._wheel_bound = False

        # Bind/unbind when pointer enters/leaves the panel
        self.interior.bind("<Enter>", self._bind_mousewheel)
        self.interior.bind("<Leave>", self._unbind_mousewheel)

    def _on_interior_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
        # Re-attach wheel tag to any new children
        if self._wheel_bound:
            self._attach_wheel_to_descendants()

    def _on_canvas_configure(self, _event=None):
        self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())

    # ---------- wheel binding (local/tag-based) ----------
    def _bind_mousewheel(self, _event=None):
        if self._wheel_bound:
            return
        self._wheel_bound = True

        # Ensure tag has our handlers
        self._ensure_panelwheel_bindings()

        # Attach tag to canvas, interior, and all descendants
        self._attach_wheel_tag(self.canvas)
        self._attach_wheel_tag(self.interior)
        self._attach_wheel_to_descendants()

    def _unbind_mousewheel(self, _event=None):
        if not self._wheel_bound:
            return
        self._wheel_bound = False
        # Tag remains but has no active handlers

    def _ensure_panelwheel_bindings(self):
        tag = "PanelWheel"
        self.canvas.bind_class(tag, "<MouseWheel>", self._on_mousewheel_osx_win, add=True)
        self.canvas.bind_class(tag, "<Shift-MouseWheel>", self._on_shiftwheel_osx_win, add=True)
        self.canvas.bind_class(tag, "<Button-4>", self._on_wheel_linux_up, add=True)
        self.canvas.bind_class(tag, "<Button-5>", self._on_wheel_linux_down, add=True)

    def _attach_wheel_to_descendants(self):
        for w in self.interior.winfo_children():
            self._attach_wheel_tag(w)
            if isinstance(w, (ttk.Frame, tk.Frame, ttk.Labelframe)):
                for c in w.winfo_children():
                    self._attach_wheel_tag(c)

    def _attach_wheel_tag(self, widget):
        tags = list(widget.bindtags())
        if "PanelWheel" not in tags:
            widget.bindtags(("PanelWheel",) + tuple(tags))

    # ---------- wheel handlers ----------
    def _on_mousewheel_osx_win(self, event):
        self._wheel_accum += event.delta
        step = 0
        while abs(self._wheel_accum) >= 120:
            step += -1 if self._wheel_accum > 0 else 1
            self._wheel_accum -= 120 * (1 if self._wheel_accum > 0 else -1)
        if step:
            self.canvas.yview_scroll(step, "units")
        return "break"

    def _on_shiftwheel_osx_win(self, event):
        direction = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(direction, "units")
        return "break"

    def _on_wheel_linux_up(self, _event):
        self.canvas.yview_scroll(-1, "units")
        return "break"

    def _on_wheel_linux_down(self, _event):
        self.canvas.yview_scroll(1, "units")
        return "break"

# ---------- Fitting utilities ----------
# ---------- File loader (CSV/TXT/DAT) ----------
def load_xy_any(path: str):
    """Wrapper around :func:`core.data_io.load_xy` for backwards compatibility."""
    return _dio.load_xy(path)




def _set_thread_limits(strategy: str, blas_threads):
    """No-op: clamping is centralized in infra.performance.blas_limit_ctx.

    We still log threadpool_info pre/post for visibility, but do not clamp here.
    """

    return


def _to_float_or_none(val):
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


# ---------- Main GUI ----------
class PeakFitApp:
    # Local copy to normalize GUI jitter slider (percent → fraction)
    def _norm_jitter_val(self, v: float) -> float:
        """Clamp bootstrap jitter to a safe [0..0.25] range."""

        try:
            v = float(v)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        return v if v <= 0.25 else 0.25

    def _is_bootstrap_selected(self) -> bool:
        try:
            key = self._unc_selected_method_key()
            return bool(key and str(key).lower() == "bootstrap")
        except Exception:
            return False

    def _update_bootstrap_visibility(self):
        if getattr(self, "_boot_opts_frame", None) is None:
            return
        (self._boot_opts_frame.grid if self._is_bootstrap_selected() else self._boot_opts_frame.grid_remove)()

    def dlog(self, msg):
        """Instance debug logger respecting the UI toggle."""
        try:
            if _safe_bool(self.debug_mode_var.get(), False):
                self.log_threadsafe(f"[DEBUG] {msg}")
        except Exception:
            pass

    # --- Bootstrap parity helpers (F) ---
    def _bootstrap_cfg_from_gui(self):
        """Extract Bootstrap + perf + band prefs mirroring GUI settings."""
        cfg = getattr(self, "cfg", {}) or {}
        solver = None
        try:
            solver = getattr(self, "boot_solver_choice", None)
            if solver is not None:
                solver = solver.get()
        except Exception:
            solver = None
        if not solver:
            solver = cfg.get("unc_boot_solver") or cfg.get("solver_choice") or "modern_vp"
        solver = str(solver)

        try:
            n_boot = int(getattr(self, "boot_n_var", None).get())  # type: ignore[arg-type]
        except Exception:
            try:
                n_boot = int(cfg.get("bootstrap_n", 1000) or 1000)
            except Exception:
                n_boot = 1000

        try:
            jitter_raw = getattr(self, "bootstrap_jitter_var", None)
            if jitter_raw is not None:
                jv = float(jitter_raw.get())
                jitter_val = jv / 100.0 if jv > 1.0 else jv
            else:
                jitter_val = float(cfg.get("bootstrap_jitter", 0.0))
        except Exception:
            jitter_val = 0.0
        jitter = self._norm_jitter_val(jitter_val)

        seed = None
        try:
            seed_all = bool(getattr(self, "perf_seed_all", None).get())  # type: ignore[arg-type]
        except Exception:
            seed_all = bool(cfg.get("perf_seed_all", False))
        if seed_all:
            seed = self._perf_seed_value()

        try:
            workers_raw = getattr(self, "perf_unc_workers", None)
            workers_val = int(float(workers_raw.get())) if workers_raw is not None else int(cfg.get("perf_unc_workers", 0) or 0)
        except Exception:
            try:
                workers_val = int(cfg.get("perf_unc_workers", 0) or 0)
            except Exception:
                workers_val = 0
        workers = workers_val if workers_val > 0 else None

        band_pref = bool(cfg.get("ui_band_pref_bootstrap", True))
        return {
            "solver": solver,
            "n": int(n_boot),
            "jitter": float(jitter),
            "workers": workers,
            "seed": seed,
            "band_pref": band_pref,
        }

    # --- small config helper used by Bootstrap controls ---
    def _cfg_set(self, key, value):
        try:
            self.cfg[key] = value
            save_config(self.cfg)
            self._debug_snapshot("config.saved", key=key, value=value)
        except Exception:
            pass

    def _persist_compute_prefs(self):
        """Persist per-method compute-band preferences."""
        prefs = {k: bool(v) for k, v in (getattr(self, "_compute_pref", {}) or {}).items()}
        try:
            self.cfg["unc_compute_pref"] = dict(prefs)
        except Exception:
            pass
        try:
            self.cfg["unc_compute_pref_json"] = json.dumps(prefs)
        except Exception:
            pass
        try:
            save_config(self.cfg)
        except Exception:
            pass

    def _persist_band_prefs(self):
        """Store per-method Show/Compute preferences in the config."""
        compute_pref = getattr(self, "_compute_pref", {}) or {}
        try:
            self.cfg["unc_band_pref"] = dict(self._band_pref)
        except Exception:
            pass
        try:
            self.cfg["unc_compute_pref"] = dict(compute_pref)
        except Exception:
            pass
        try:
            self.cfg["unc_compute_pref_json"] = json.dumps(
                {k: bool(v) for k, v in compute_pref.items()}
            )
        except Exception:
            pass
        try:
            self.cfg["ui_band_pref_asymptotic"] = bool(
                self._band_pref.get("asymptotic", False)
            )
            self.cfg["ui_band_pref_bootstrap"] = bool(
                self._band_pref.get("bootstrap", True)
            )
            self.cfg["ui_band_pref_bayesian"] = bool(
                self._band_pref.get("bayesian", False)
            )
            self.cfg["ui_band_compute_asymptotic"] = bool(
                compute_pref.get("asymptotic", True)
            )
            self.cfg["ui_band_compute_bootstrap"] = bool(
                compute_pref.get("bootstrap", True)
            )
            self.cfg["ui_band_compute_bayesian"] = bool(
                compute_pref.get("bayesian", True)
            )
            save_config(self.cfg)
        except Exception:
            pass

    def _perf_seed_value(self) -> Optional[int]:
        seed_cfg = self.cfg.get("perf_seed", "")
        seed_var = getattr(self, "seed_var", None)
        if seed_var is not None:
            try:
                seed_str = str(seed_var.get()).strip()
            except Exception:
                seed_str = ""
        else:
            try:
                seed_str = str(seed_cfg).strip()
            except Exception:
                seed_str = ""
        if seed_str in ("", "None"):
            return None
        try:
            return int(seed_str)
        except Exception:
            return None

    @log_action
    def _on_debug_toggle(self, *_):
        try:
            self._cfg_set("log_debug_mode", bool(self.debug_mode_var.get()))
        except Exception:
            pass

    def _select_bootstrap_solver(self) -> str:
        main = (
            self.solver_choice.get()
            if hasattr(self, "solver_choice")
            else "modern_trf"
        )
        boot: Optional[str] = None
        if hasattr(self, "boot_solver_choice"):
            try:
                boot = self.boot_solver_choice.get()
            except Exception:
                boot = None
        if not boot:
            boot = self.cfg.get("unc_boot_solver") or main
        try:
            if str(boot).lower().startswith("lmfit") and not bool(
                getattr(self, "has_lmfit", False)
            ):
                boot = "modern_trf"
                if hasattr(self, "boot_solver_choice"):
                    try:
                        self.boot_solver_choice.set(boot)
                    except Exception:
                        pass
                try:
                    self._cfg_set("unc_boot_solver", boot)
                except Exception:
                    pass
        except Exception:
            pass
        return str(boot or main)

    def __init__(self, root, cfg=None):
        self.root = root
        # The Tk thread, and what worker threads hand it (see run_in_thread).
        self._tk_thread = threading.current_thread()
        self._ui_calls = queue.SimpleQueue()
        self._log_lock = threading.Lock()
        # The abort event of each running Run, Export and Run Batch (see _job_begin);
        # closing the window waits for them (see _on_close).
        self._jobs: set = set()
        self._closing = False
        self.cfg = cfg if cfg is not None else load_config()
        self.cfg.setdefault("baseline_uses_fit_range", True)
        self.cfg.setdefault("ui_show_uncertainty_band", False)
        self.cfg.setdefault("perf_numba", False)
        self.cfg.setdefault("perf_cache_baseline", True)
        self.cfg.setdefault("perf_seed_all", False)
        self.cfg.setdefault("perf_seed", "")
        self.cfg.setdefault("perf_unc_workers", 0)
        self.cfg.setdefault("perf_parallel_strategy", "outer")
        self.cfg.setdefault("perf_blas_threads", 0)
        self.cfg.setdefault("last_template_name", self.cfg.get("auto_apply_template_name", ""))
        self.cfg.setdefault("export_unc_wide", False)
        self.cfg.setdefault("unc_alpha", 0.05)
        self.cfg.setdefault("unc_center_resid", True)
        # --- Per-method CI-band preference (state sync) ---
        self.cfg.setdefault("ui_band_pref_asymptotic", False)
        self.cfg.setdefault("ui_band_pref_bootstrap", True)
        self.cfg.setdefault("ui_band_pref_bayesian", False)
        self.cfg.setdefault("ui_band_compute_asymptotic", True)
        self.cfg.setdefault("ui_band_compute_bootstrap", True)
        self.cfg.setdefault("ui_band_compute_bayesian", True)
        self.cfg.setdefault("unc_band_pref", {})
        self.cfg.setdefault("unc_compute_pref", {})
        stored_band_pref: Dict[str, bool] = {}
        try:
            stored_band_pref.update({k: bool(v) for k, v in dict(self.cfg.get("unc_band_pref", {})).items()})
        except Exception:
            pass
        stored_compute_pref: Dict[str, bool] = {}
        try:
            stored_compute_pref.update({k: bool(v) for k, v in dict(self.cfg.get("unc_compute_pref", {})).items()})
        except Exception:
            pass
        try:
            raw_json = self.cfg.get("unc_compute_pref_json", "")
            if raw_json:
                stored_compute_pref.update({k: bool(v) for k, v in json.loads(raw_json).items()})
        except Exception:
            pass
        self._band_pref = {
            "asymptotic": bool(self.cfg.get("ui_band_pref_asymptotic", False)),
            "bootstrap": bool(self.cfg.get("ui_band_pref_bootstrap", True)),
            "bayesian": bool(self.cfg.get("ui_band_pref_bayesian", False)),
        }
        self._band_pref.update(stored_band_pref)
        self._compute_pref = {
            "asymptotic": bool(self.cfg.get("ui_band_compute_asymptotic", True)),
            "bootstrap": bool(self.cfg.get("ui_band_compute_bootstrap", True)),
            "bayesian": bool(self.cfg.get("ui_band_compute_bayesian", True)),
        }
        self._compute_pref.update(stored_compute_pref)
        self._last_unc_method_key = None  # tracks source method for band-pref memory
        self._unc_change_busy = False
        # Group B: user override marker for bootstrap solver selection
        self.boot_solver_overridden = False
        self.cfg.setdefault("bootstrap_jitter", 0.0)
        self.cfg.setdefault("log_debug_mode", False)
        self.cfg.setdefault("bayes_walkers", 0)
        self.cfg.setdefault("bayes_burn", 1000)
        self.cfg.setdefault("bayes_steps", 8000)
        self.cfg.setdefault("bayes_thin", 1)
        self.cfg.setdefault("bayes_prior_sigma", "half_cauchy")
        # Bayesian band + diagnostics defaults (persisted for cache parity)
        self.cfg.setdefault("bayes_band_enabled", False)
        self.cfg.setdefault("bayes_band_force", False)
        self.cfg.setdefault("bayes_band_max_draws", 512)
        self.cfg.setdefault("bayes_diagnostics", False)
        self.cfg.setdefault("bayes_diag_ess_min", 200.0)
        self.cfg.setdefault("bayes_diag_rhat_max", 1.05)
        self.cfg.setdefault("bayes_diag_mcse_mean", float("inf"))
        # Debug/log defaults so debug calls are safe before UI builds
        self._log_console = None
        self._log_visible = False
        self._log_frame = None
        self._log_buffer: list[str] = []
        self._auto_show_log_on_warn = True
        self._auto_show_log_on_error = True
        if "bootstrap_seed" in self.cfg:
            try:
                self.log_threadsafe("Note: 'bootstrap_seed' is deprecated; use Performance → Seed.")
            except Exception:
                pass
        save_config(self.cfg)
        self.root.title("Interactive Peak Fit (pseudo-Voigt)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._style = getattr(self, "_style", ttk.Style(self.root))
        # Configure buttons with better text rendering (antialiasing support)
        self._style.configure("Danger.TButton", foreground="white", background="#c62828")
        self._style.map("Danger.TButton", background=[("active", "#b71c1c")])
        self._style.configure("Success.TButton", foreground="white", background="#2e7d32")
        self._style.map("Success.TButton", background=[("active", "#1b5e20")])

        performance.set_logger(self.log_threadsafe)

        try:
            base = tkfont.nametofont("TkDefaultFont")
            self.default_font = base
            self._bold_font = base.copy()
            self._bold_font.configure(weight="bold")
        except Exception:
            # Fallback to selected sans family or default size if font selection failed
            from ui.fonts import _selected_sans, DEFAULT_UI_SIZE
            fallback_family = _selected_sans if _selected_sans else "Helvetica"
            self.default_font = tkfont.Font(family=fallback_family, size=DEFAULT_UI_SIZE)
            self._bold_font = self.default_font.copy()
            self._bold_font.configure(weight="bold")

        last_template = self.cfg.get("last_template_name", "")

        # Data
        self.x = None
        self.y_raw = None
        self.baseline = None
        self.use_baseline = tk.BooleanVar(value=True)

        # Baseline mode: "add" (fit over baseline) or "subtract"
        self.baseline_mode = tk.StringVar(value="add")
        # Option: compute ALS baseline only from fit range
        self.baseline_use_range = tk.BooleanVar(value=bool(self.cfg.get("baseline_uses_fit_range", True)))
        self.baseline_use_range.trace_add("write", self.on_baseline_use_range_toggle)

        bd = self.cfg.get("baseline_defaults", {})
        als_def = bd.get("als", {})
        poly_def = bd.get("polynomial", {})
        self.base_method_var = tk.StringVar(value=bd.get("method", "als"))
        self.als_lam = tk.DoubleVar(value=als_def.get("lam", 1e5))
        self.als_asym = tk.DoubleVar(value=als_def.get("p", 0.001))
        self.als_niter = tk.IntVar(value=als_def.get("niter", 10))
        self.als_thresh = tk.DoubleVar(value=als_def.get("thresh", 0.0))
        self.poly_degree_var = tk.IntVar(value=poly_def.get("degree", 2))
        self.poly_norm_var = tk.BooleanVar(value=poly_def.get("normalize_x", True))
        self.global_eta = tk.DoubleVar(value=self.cfg.get("ui_eta", 0.5))
        self.global_eta.trace_add("write", lambda *_: self._on_eta_change())
        self.auto_apply_template = tk.BooleanVar(value=bool(self.cfg.get("auto_apply_template", False)))
        self.auto_apply_template_name = tk.StringVar(value=last_template)

        # Interaction
        self.add_peaks_mode = tk.BooleanVar(value=bool(self.cfg.get("ui_add_peaks_on_click", True)))

        # Fit range (None = full)
        self.fit_xmin: Optional[float] = None
        self.fit_xmax: Optional[float] = None
        self.fit_min_var = tk.StringVar(value="")
        self.fit_max_var = tk.StringVar(value="")

        # Span/interaction state
        self._span = None
        self._span_active = False
        self._span_prev_click_toggle = None
        self._span_cids: list[int] = []
        self._cursor_before_span: str = ""

        # Peaks
        self.peaks: List[Peak] = []

        # Templates UI state
        self.template_var = tk.StringVar(value=last_template)

        # Components visibility
        self.components_visible = bool(self.cfg.get("ui_show_components", True))

        # Matplotlib click binding
        self.cid = None

        # Axis label
        self.x_label_var = tk.StringVar(value=str(self.cfg.get("x_label", "x")))
        self.x_label_auto_math = tk.BooleanVar(value=bool(self.cfg.get("x_label_auto_math", True)))
        self.show_legend_var = tk.BooleanVar(value=bool(self.cfg.get("ui_show_legend", True)))
        self.legend_center_sigfigs = tk.IntVar(value=int(self.cfg.get("legend_center_sigfigs", 6)))

        # Batch defaults
        self.batch_patterns = tk.StringVar(value=self.cfg.get("batch_patterns", "*.csv;*.txt;*.dat"))
        self.batch_source = tk.StringVar(value=self.cfg.get("batch_source", "template"))
        self.batch_reheight = tk.BooleanVar(value=bool(self.cfg.get("batch_reheight", False)))
        self.batch_auto_max = tk.IntVar(value=int(self.cfg.get("batch_auto_max", 5)))
        self.batch_save_traces = tk.BooleanVar(value=bool(self.cfg.get("batch_save_traces", False)))
        self.batch_unc_enabled = tk.BooleanVar(value=bool(self.cfg.get("batch_compute_uncertainty", False)))

        self._baseline_cache = {}

        # Solver selection and diagnostics
        try:
            import lmfit  # noqa: F401
            self.has_lmfit = True
        except Exception:
            self.has_lmfit = False
        self.solver_choice = tk.StringVar(value=self.cfg.get("solver_choice", "modern_vp"))
        if not self.has_lmfit and self.solver_choice.get() == "lmfit_vp":
            self.solver_choice.set("modern_vp")
            self.cfg["solver_choice"] = "modern_vp"
            save_config(self.cfg)
        # Group B: legacy follow config removed and bootstrap solver defaults to main solver
        self.cfg.pop("unc_boot_solver_follow", None)
        self.cfg.setdefault("unc_boot_solver", self.solver_choice.get())
        _base_key = self.solver_choice.get()
        self.solver_title = tk.StringVar(value=SOLVER_LABELS.get(_base_key, _base_key))
        self.classic_maxfev = tk.IntVar(value=20000)
        self.classic_centers_window = tk.BooleanVar(value=True)
        self.classic_margin = tk.DoubleVar(value=0.0)
        self.classic_fwhm_min = tk.DoubleVar(value=0.5)
        self.classic_fwhm_max = tk.DoubleVar(value=2.0)
        self.classic_height_factor = tk.DoubleVar(value=1.0)
        self.modern_loss = tk.StringVar(value="linear")
        self.modern_weight = tk.StringVar(value="none")
        self.modern_fscale = tk.DoubleVar(value=1.0)
        self.modern_maxfev = tk.IntVar(value=20000)
        self.modern_restarts = tk.IntVar(value=1)
        self.modern_jitter = tk.DoubleVar(value=0.0)
        self.modern_centers_window = tk.BooleanVar(value=True)
        self.modern_min_fwhm = tk.BooleanVar(value=True)
        self.lmfit_algo = tk.StringVar(value="least_squares")
        self.lmfit_maxfev = tk.IntVar(value=20000)
        self.cfg.setdefault("lmfit_share_fwhm", False)
        self.cfg.setdefault("lmfit_share_eta", False)
        self.lmfit_share_fwhm = tk.BooleanVar(value=bool(self.cfg.get("lmfit_share_fwhm", False)))
        self.lmfit_share_eta = tk.BooleanVar(value=bool(self.cfg.get("lmfit_share_eta", False)))
        self.lmfit_share_fwhm.trace_add(
            "write",
            lambda *_: self._cfg_set("lmfit_share_fwhm", bool(self.lmfit_share_fwhm.get())),
        )
        self.lmfit_share_eta.trace_add(
            "write",
            lambda *_: self._cfg_set("lmfit_share_eta", bool(self.lmfit_share_eta.get())),
        )
        self.snr_text = tk.StringVar(value="S/N: --")

        self.debug_mode_var = tk.BooleanVar(value=bool(self.cfg.get("log_debug_mode", False)))
        self.perf_parallel_strategy = tk.StringVar(
            value=str(self.cfg.get("perf_parallel_strategy", "outer"))
        )
        self.perf_blas_threads = tk.IntVar(value=int(self.cfg.get("perf_blas_threads", 0)))
        self.debug_mode_var.trace_add("write", lambda *_: self._on_debug_toggle())

        # Debug logging convenience
        self._debug_prefix = "[DEBUG]"
        self._debug_snapshot("config.loaded")

        unc_section_cfg = dict(self.cfg.get("uncertainty") or {})
        return_band_pref = bool(
            unc_section_cfg.get(
                "return_band",
                self.cfg.get("ui_show_uncertainty_band", False),
            )
        )
        compute_band_pref = bool(unc_section_cfg.get("compute_band", True))
        self.ci_band = None
        self.show_ci_band = tk.BooleanVar(value=return_band_pref)
        self.show_ci_band_var = self.show_ci_band
        try:
            existing_compute_var = self.compute_ci_band_var  # type: ignore[attr-defined]
        except AttributeError:
            existing_compute_var = None
        if isinstance(existing_compute_var, tk.BooleanVar):
            try:
                existing_compute_var.set(bool(existing_compute_var.get()))
            except Exception:
                pass
        else:
            try:
                self.compute_ci_band_var = tk.BooleanVar(value=compute_band_pref)
            except Exception:
                self.compute_ci_band_var = None
        if isinstance(getattr(self, "compute_ci_band_var", None), tk.BooleanVar):
            try:
                self.compute_ci_band_var.set(bool(compute_band_pref))
            except Exception:
                pass
        self._compute_chk = None
        # The last fit and what it was fitted to (see _current_fit), and the
        # uncertainty computed for it (see _cached_uncertainty)
        self._fit = None
        self._unc_cache = None
        self._export_running = False
        # Uncertainty state
        self.last_uncertainty = None
        self._last_unc_locks: List[Dict[str, bool]] = []

        self.current_file: Optional[Path] = None

        # Uncertainty and performance controls
        unc_cfg = unc_section_cfg.get("method", self.cfg.get("unc_method", "asymptotic"))
        if unc_cfg == "bootstrap":
            _base_key = self.solver_choice.get()
            _base_label = SOLVER_LABELS.get(_base_key, _base_key)
            unc_label = f"Bootstrap (base solver = {_base_label})"
        elif unc_cfg == "bayesian":
            unc_label = "Bayesian"
        else:
            unc_label = "Asymptotic"
        self.unc_method = tk.StringVar(value=unc_label)
        self.unc_method_var = self.unc_method
        # ensure band toggle reflects current method on startup
        try:
            self._on_uncertainty_method_changed()
        except Exception:
            pass
        # Performance apply scheduler/debounce infrastructure (for spam reduction).
        # Defined before the traces below, which use it.
        self._perf_apply_after_id: Optional[str] = None
        self._perf_last_signature: Optional[tuple] = None
        self._perf_pending_source: str = ""
        # No GPU switch, GPU chunk or fit-worker count (game plan 3.3): CuPy made the
        # fits 2.6 to 7x slower, nothing reads the chunk, and nothing fits in parallel.
        self.perf_numba = tk.BooleanVar(value=bool(self.cfg.get("perf_numba", False)))
        self.perf_cache_baseline = tk.BooleanVar(value=bool(self.cfg.get("perf_cache_baseline", True)))
        self.perf_seed_all = tk.BooleanVar(value=bool(self.cfg.get("perf_seed_all", False)))
        self.perf_unc_workers = tk.IntVar(value=int(self.cfg.get("perf_unc_workers", 0)))
        self.perf_numba.trace_add("write", lambda *_: self.request_apply_performance(source="trace:perf_numba"))
        self.perf_cache_baseline.trace_add("write", lambda *_: self.request_apply_performance(source="trace:perf_cache_baseline"))
        self.perf_seed_all.trace_add("write", lambda *_: self.request_apply_performance(source="trace:perf_seed_all"))
        self.perf_unc_workers.trace_add("write", lambda *_: self.request_apply_performance(source="trace:perf_unc_workers"))
        self.alpha_var = tk.DoubleVar(value=float(self.cfg.get("unc_alpha", 0.05)))
        self.center_resid_var = tk.BooleanVar(value=bool(self.cfg.get("unc_center_resid", True)))
        self.bootstrap_jitter_var = tk.DoubleVar(value=100.0 * float(self.cfg.get("bootstrap_jitter", 0.0)))

        self.bayes_walkers_var = tk.IntVar(value=int(self.cfg.get("bayes_walkers", 0)))  # 0 => auto
        self.bayes_burn_var = tk.IntVar(value=int(self.cfg.get("bayes_burn", 1000)))
        self.bayes_steps_var = tk.IntVar(value=int(self.cfg.get("bayes_steps", 8000)))
        self.bayes_thin_var = tk.IntVar(value=int(self.cfg.get("bayes_thin", 1)))
        self.bayes_prior_var = tk.StringVar(value=str(self.cfg.get("bayes_prior_sigma", "half_cauchy")))
        def _on_prior_change(*_e):
            try:
                self._cfg_set("bayes_prior_sigma", str(self.bayes_prior_var.get()))
            except Exception:
                pass

        try:
            self.bayes_prior_var.trace_add("write", _on_prior_change)
        except Exception:
            pass
        self.bayes_diag_var = tk.BooleanVar(
            value=bool(
                self.cfg.get("bayes_diagnostics", self.cfg.get("bayes_diag", False))
            )
        )
        # --- Bayesian band + diagnostics GUI state ---
        self.bayes_band_enabled_var = tk.BooleanVar(value=bool(self.cfg.get("bayes_band_enabled", False)))
        self.bayes_band_force_var = tk.BooleanVar(value=bool(self.cfg.get("bayes_band_force", False)))
        self.bayes_band_max_draws_var = tk.IntVar(value=int(self.cfg.get("bayes_band_max_draws", 512)))
        self.bayes_thresh_ess_min_var = tk.DoubleVar(value=float(self.cfg.get("bayes_diag_ess_min", 200.0)))
        self.bayes_thresh_rhat_max_var = tk.DoubleVar(value=float(self.cfg.get("bayes_diag_rhat_max", 1.05)))
        _mcse_cfg = self.cfg.get("bayes_diag_mcse_mean", float("inf"))
        self.bayes_thresh_mcse_mean_str = tk.StringVar(
            value=("" if not np.isfinite(_mcse_cfg) else str(_mcse_cfg))
        )
        self.seed_var = tk.StringVar(value=str(self.cfg.get("perf_seed", "")))
        self.seed_var.trace_add("write", lambda *_: self.request_apply_performance(source="trace:seed_var"))

        # UI
        self._build_ui()
        self._new_figure()
        self._update_template_info()
        self.request_apply_performance(source="startup", immediate=True, save_cfg=False)

        # Uncertainty job tracking / last log
        self._unc_job_id = 0
        self._unc_running = False
        self._last_unc_log = None

        # what loading the settings changed in an older file, said once
        while _SETTINGS_NOTES:
            self.status_warn(_SETTINGS_NOTES.pop(0))

        self._debug_snapshot("init")

    # --- Debug API ----------------------------------------------------------
    def _is_debug(self) -> bool:
        try:
            return bool(self.debug_mode_var.get())
        except Exception:
            return bool(getattr(self, "cfg", {}).get("log_debug_mode", False))

    def _dbg(self, msg: str) -> None:
        if not self._is_debug():
            return
        try:
            self.log(f"{self._debug_prefix} {msg}")
        except Exception:
            pass

    def _debug_snapshot(self, where: str, **extra) -> None:
        if not self._is_debug():
            return
        try:
            label = str(self.unc_method.get()) if hasattr(self, "unc_method") else ""
        except Exception:
            label = ""
        try:
            key = method_display_to_key(label)
        except Exception:
            key = str(label).strip().lower()
        try:
            show_band = bool(self.show_ci_band_var.get()) if hasattr(self, "show_ci_band_var") else False
        except Exception:
            show_band = False
        try:
            compute_band = self._should_compute_ci_band()
        except Exception:
            compute_band = False
        try:
            seed_all = bool(self.perf_seed_all.get()) if hasattr(self, "perf_seed_all") else False
        except Exception:
            seed_all = False
        try:
            seed_val = self._perf_seed_value() if seed_all and hasattr(self, "_perf_seed_value") else None
        except Exception:
            seed_val = None
        try:
            workers_unc = pipeline.uncertainty_workers({"perf_unc_workers": self.perf_unc_workers.get()})[0]
        except Exception:
            workers_unc = None
        state = {
            "file": getattr(getattr(self, "current_file", None), "name", "—"),
            "peaks": len(getattr(self, "peaks", []) or []),
            "unc_label": label,
            "unc_key": key,
            "solver": str(self.solver_choice.get()) if hasattr(self, "solver_choice") else "",
            "show_band": show_band,
            "compute_band": compute_band,
            "seed_all": seed_all,
            "seed": seed_val,
            "workers_unc": workers_unc,
        }
        state.update(extra or {})
        self._dbg(f"{where}: {_kv(state)}")

    def _ensure_compute_toggle(self):
        """Add a 'Compute band' checkbox just under the 'Show band' checkbox."""
        try:
            if self._compute_chk is not None or self.compute_ci_band_var is None:
                return
            parent = getattr(self, "_ci_toggle_chk", None)
            parent = parent.master if parent is not None else None
            if parent is None:
                return
            import tkinter.ttk as ttk

            self._compute_chk = ttk.Checkbutton(
                parent,
                text="Compute band",
                variable=self.compute_ci_band_var,
                command=self._toggle_ci_compute,
            )
            self._compute_chk.pack(anchor="w", padx=4, pady=(0, 2))
        except Exception:
            pass

    # ----- UI -----
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        top.columnconfigure(0, weight=1)

        self.action_bar = ttk.Frame(top)
        self.action_bar.grid(row=0, column=0, sticky="w")

        self.file_label = ttk.Label(top, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky="e")

        file_seg = ttk.Frame(self.action_bar)
        file_seg.pack(side=tk.LEFT)
        ttk.Button(file_seg, text="Open", command=self.on_open).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_seg, text="Export", command=self.on_export).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_seg, text="Batch", command=self.run_batch).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self.action_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)

        fit_seg = ttk.Frame(self.action_bar)
        fit_seg.pack(side=tk.LEFT)
        try:
            self.step_icon = tk.PhotoImage(data=STEP_ICON_B64)
            # Draw the icon in the button text colour, so it also shows on a dark theme
            paint_icon(self.step_icon, self._style.lookup("TButton", "foreground"))
            self.step_btn = ttk.Button(fit_seg, image=self.step_icon, command=self.step_once)
        except Exception:
            self.step_btn = ttk.Button(fit_seg, text="Step", command=self.step_once)
        self.step_btn.pack(side=tk.LEFT, padx=2)
        self.fit_btn = ttk.Button(fit_seg, text="Fit", command=self.fit)
        self.fit_btn.pack(side=tk.LEFT, padx=2)
        # Apply bold style for Fit button (configured via configure_ttk_styles in main())
        self._style = getattr(self, "_style", ttk.Style())
        self.fit_btn.configure(style="Fit.TButton")

        ttk.Separator(self.action_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)

        graph_seg = ttk.Frame(self.action_bar)
        graph_seg.pack(side=tk.LEFT)
        ttk.Button(
            graph_seg,
            text="Uncertainty",
            command=lambda: self.show_ci_band_var.set(not self.show_ci_band_var.get()),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(graph_seg, text="Legend", command=self._toggle_legend_action).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(graph_seg, text="Components", command=self.toggle_components).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(self.action_bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=3
        )

        help_seg = ttk.Frame(self.action_bar)
        help_seg.pack(side=tk.LEFT)
        ttk.Button(help_seg, text="Help", command=self.show_help).pack(
            side=tk.LEFT, padx=2
        )

        self.root.bind("<F1>", lambda e: self.show_help())

        self._update_file_label(None)

        mid = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(mid)
        right_wrapper = ttk.Frame(mid)

        mid.add(left, weight=3)
        mid.add(right_wrapper, weight=1)

        # sensible limits
        self._min_left_px = 480
        self._max_right_px = 560

        def _clamp_sash(_evt=None):
            try:
                total = mid.winfo_width()
                if total <= 1:
                    return
                pos = mid.sashpos(0)
                lo = self._min_left_px
                hi = max(lo + 100, total - self._max_right_px)
                pos = max(lo, min(pos, hi))
                mid.sashpos(0, pos)
            except Exception:
                pass

        mid.bind("<B1-Motion>", _clamp_sash)
        mid.bind("<Configure>", _clamp_sash)
        self.root.after(0, _clamp_sash)

        # Left: plot
        self.fig = plt.Figure(figsize=(7,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(format_axis_label_inline(self.x_label_var.get(), self.x_label_auto_math.get()))
        self.ax.set_ylabel("Intensity")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.nav = NavigationToolbar2Tk(self.canvas, left)
        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click_plot)

        # Right: scrollable controls
        right_scroll = ScrollableFrame(right_wrapper)
        right_scroll.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.right_scroll = right_scroll
        right = right_scroll.interior
        try:
            right.configure(padding=6)
        except Exception:
            pass

        # Baseline box
        baseline_box = ttk.Labelframe(right, text="Baseline"); baseline_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(baseline_box, text="Apply baseline", variable=self.use_baseline, command=self.refresh_plot).pack(anchor="w")

        method_row = ttk.Frame(baseline_box); method_row.pack(fill=tk.X, pady=2)
        ttk.Label(method_row, text="Method:").pack(side=tk.LEFT)
        method_cb = ttk.Combobox(
            method_row,
            textvariable=self.base_method_var,
            values=["als", "polynomial"],
            state="readonly",
            width=12,
        )
        method_cb.pack(side=tk.LEFT)
        method_cb.bind("<<ComboboxSelected>>", self._on_base_method_change)
        add_tooltip(
            method_cb,
            "Choose baseline method:\n"
            "- ALS: Smooth, asymmetric penalization (λ, p, iterations, threshold)\n"
            "- Polynomial: Weighted LS polynomial (degree, optional x-normalization)\n"
            "Tip: Use 'Save as default' to persist your choice."
        )

        mode_row = ttk.Frame(baseline_box); mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Add to fit", variable=self.baseline_mode, value="add", command=self.refresh_plot).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Subtract",  variable=self.baseline_mode, value="subtract", command=self.refresh_plot).pack(side=tk.LEFT, padx=4)

        _chk_range = ttk.Checkbutton(baseline_box, text="Baseline uses fit range", variable=self.baseline_use_range)
        _chk_range.pack(anchor="w", pady=(2,0))
        add_tooltip(
            _chk_range,
            "When enabled, baseline is estimated only within the current fit window.\n"
            "Outside the window, the result is smoothly extended."
        )

        self.als_frame = ttk.Frame(baseline_box)
        row = ttk.Frame(self.als_frame); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="λ (smooth):").pack(side=tk.LEFT)
        ttk.Entry(row, width=10, textvariable=self.als_lam).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="p (asym):").pack(side=tk.LEFT)
        ttk.Entry(row, width=10, textvariable=self.als_asym).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(self.als_frame); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Iterations:").pack(side=tk.LEFT)
        ttk.Entry(row2, width=5, textvariable=self.als_niter).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="Threshold:").pack(side=tk.LEFT)
        ttk.Entry(row2, width=7, textvariable=self.als_thresh).pack(side=tk.LEFT, padx=4)

        self.poly_frame = ttk.Frame(baseline_box)
        ttk.Label(self.poly_frame, text="Degree:").pack(side=tk.LEFT)
        _poly_deg = ttk.Spinbox(self.poly_frame, from_=0, to=10, width=5, textvariable=self.poly_degree_var)
        _poly_deg.pack(side=tk.LEFT, padx=4)
        add_tooltip(
            _poly_deg,
            "Polynomial degree.\n"
            "If the fit range has too few points, the effective degree is clamped to N-1.\n"
            "The UI will auto-reduce and display a status message."
        )
        _poly_norm = ttk.Checkbutton(self.poly_frame, text="Normalize x to [-1,1]", variable=self.poly_norm_var)
        _poly_norm.pack(side=tk.LEFT, padx=4)
        add_tooltip(
            _poly_norm,
            "Improves numerical conditioning by mapping x to [-1, 1] for the fit."
        )

        self._update_baseline_frames()

        ttk.Button(baseline_box, text="Recompute baseline", command=self.compute_baseline).pack(side=tk.LEFT, pady=2)
        _save_default_btn = ttk.Button(baseline_box, text="Save as default", command=self.save_baseline_default)
        _save_default_btn.pack(side=tk.LEFT, padx=4)
        add_tooltip(
            _save_default_btn,
            "Persist the current baseline method and its parameters as the default for future sessions."
        )
        ttk.Label(baseline_box, textvariable=self.snr_text).pack(side=tk.LEFT, padx=8)

        # Eta box
        eta_box = ttk.Labelframe(right, text="Shape factor η (0=Gaussian, 1=Lorentzian)"); eta_box.pack(fill=tk.X, pady=4)
        ttk.Entry(eta_box, width=10, textvariable=self.global_eta).pack(side=tk.LEFT, padx=4)
        ttk.Button(eta_box, text="Apply to all peaks", command=self.apply_eta_all).pack(side=tk.LEFT, padx=4)

        # Peaks table
        peaks_box = ttk.Labelframe(right, text="Peaks"); peaks_box.pack(fill=tk.BOTH, expand=True, pady=4)
        cols = ("idx","center","height","fwhm","lockw","lockc")
        self.tree = ttk.Treeview(peaks_box, columns=cols, show="headings", selectmode="browse")
        headers = ["#", "Center", "Height", "FWHM", "Lock W", "Lock C"]
        widths  = [30,   90,       90,       90,      70,       70]
        for c, txt, w in zip(cols, headers, widths):
            self.tree.heading(c, text=txt); self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.X, expand=False)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_peak)

        # Edit panel
        edit = ttk.Frame(peaks_box); edit.pack(fill=tk.X, pady=4)
        self.center_var = tk.DoubleVar(); self.height_var = tk.DoubleVar()
        self.fwhm_var = tk.DoubleVar()
        self.lockw_var = tk.BooleanVar()
        self.lockc_var = tk.BooleanVar()
        # NEW: width cap widgets
        self.fwhm_cap_var = tk.DoubleVar()
        self.fwhm_cap_on  = tk.BooleanVar()
        # NEW: universal apply toggle
        self.fwhm_cap_all = tk.BooleanVar()
        # Initialize from persisted config
        try:
            self.fwhm_cap_all.set(bool(self.cfg.get("ui_fwhm_cap_all", False)))
            self.fwhm_cap_var.set(float(self.cfg.get("ui_fwhm_cap_value", 50.0)))
        except Exception:
            self.fwhm_cap_all.set(False)
            self.fwhm_cap_var.set(50.0)

        e_center = ttk.Entry(edit, textvariable=self.center_var, width=10)
        e_height = ttk.Entry(edit, textvariable=self.height_var, width=10)
        e_fwhm   = ttk.Entry(edit, textvariable=self.fwhm_var,   width=10)
        ttk.Label(edit, text="Center").grid(row=0, column=0, sticky="e")
        e_center.grid(row=0, column=1, padx=2)
        ttk.Label(edit, text="Height").grid(row=1, column=0, sticky="e")
        e_height.grid(row=1, column=1, padx=2)
        ttk.Label(edit, text="FWHM").grid(row=2, column=0, sticky="e")
        e_fwhm.grid(row=2, column=1, padx=2)
        # NEW: “Max FWHM” + toggle (effective only when width is free)
        ttk.Label(edit, text="Max FWHM").grid(row=2, column=2, sticky="e")
        self.ent_fwhm_cap = ttk.Entry(edit, textvariable=self.fwhm_cap_var, width=10)
        self.ent_fwhm_cap.grid(row=2, column=3, padx=2)
        ttk.Checkbutton(edit, text="Cap", variable=self.fwhm_cap_on,
                        command=self.on_cap_toggle).grid(row=2, column=4, sticky="w")
        ttk.Checkbutton(edit, text="All", variable=self.fwhm_cap_all,
                        command=self.on_cap_toggle).grid(row=2, column=5, sticky="w")

        ttk.Checkbutton(edit, text="Lock width",  variable=self.lockw_var, command=self.on_lock_toggle).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Checkbutton(edit, text="Lock center", variable=self.lockc_var, command=self.on_lock_toggle).grid(row=3, column=1, sticky="w", pady=2)

        btns = ttk.Frame(peaks_box); btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="➕ Add Peak", command=self.add_peak_from_fields).pack(side=tk.LEFT)
        ttk.Button(btns, text="Apply edits", command=self.apply_edits).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete selected", command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear all", command=self.clear_peaks).pack(side=tk.LEFT, padx=4)

        for w in (e_center, e_height, e_fwhm):
            w.bind("<Return>", lambda _e: self.apply_edits())
        # Enter on cap entry: apply to all if "All" is enabled, else edit selected; persist to cfg
        self.ent_fwhm_cap.bind("<Return>", lambda _e: self._commit_cap_entry())
        # Also persist on focus-out so value sticks even without toggling
        self.ent_fwhm_cap.bind("<FocusOut>", lambda _e: self._sync_cap_config())

        # Interaction
        inter = ttk.Labelframe(right, text="Interaction"); inter.pack(fill=tk.X, pady=6)
        self.add_peaks_checkbox = ttk.Checkbutton(
            inter,
            text="Add peaks on click",
            variable=self.add_peaks_mode,
            command=self._on_add_peaks_toggle,
        )
        self.add_peaks_checkbox.pack(anchor="w")
        ttk.Label(inter, text="Tip: while toolbar Zoom/Pan is active, clicks never add peaks.").pack(anchor="w")
        row_zoom = ttk.Frame(inter); row_zoom.pack(fill=tk.X, pady=(4,0))
        ttk.Button(row_zoom, text="Zoom out", command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Button(row_zoom, text="Reset view", command=self.reset_view).pack(side=tk.LEFT, padx=4)

        # Fit range
        fr = ttk.Labelframe(right, text="Fit range (x)"); fr.pack(fill=tk.X, pady=6)
        ttk.Label(fr, text="Min").grid(row=0, column=0, sticky="e")
        ttk.Entry(fr, width=10, textvariable=self.fit_min_var).grid(row=0, column=1, padx=2)
        ttk.Label(fr, text="Max").grid(row=0, column=2, sticky="e")
        ttk.Entry(fr, width=10, textvariable=self.fit_max_var).grid(row=0, column=3, padx=2)
        ttk.Button(fr, text="Apply", command=self.apply_fit_range_from_fields).grid(row=0, column=4, padx=4)
        ttk.Button(fr, text="Select on plot", command=self.enable_span).grid(row=1, column=1, columnspan=2, pady=2)
        ttk.Button(fr, text="Full range", command=self.clear_fit_range).grid(row=1, column=3, pady=2)

        # Templates
        tmpl = ttk.Labelframe(right, text="Peak Templates"); tmpl.pack(fill=tk.X, pady=6)
        self.template_info = ttk.Label(tmpl, text="Templates: 0")
        self.template_info.pack(anchor="w", pady=(0,2))
        rowt = ttk.Frame(tmpl); rowt.pack(fill=tk.X)
        ttk.Label(rowt, text="Select").pack(side=tk.LEFT)
        self.template_combo = ttk.Combobox(rowt, textvariable=self.template_var, state="readonly", width=18, values=[])
        self.template_combo.pack(side=tk.LEFT, padx=4)
        self.template_combo.bind("<<ComboboxSelected>>", lambda e: self._persist_last_template(self.template_var.get()))
        ttk.Button(rowt, text="Apply", command=self.apply_selected_template).pack(side=tk.LEFT, padx=2)

        rowt2 = ttk.Frame(tmpl); rowt2.pack(fill=tk.X, pady=2)
        ttk.Button(rowt2, text="Save as new…", command=self.save_template_as).pack(side=tk.LEFT)
        ttk.Button(rowt2, text="Save changes", command=self.save_changes_to_selected_template).pack(side=tk.LEFT, padx=4)
        ttk.Button(rowt2, text="Delete", command=self.delete_selected_template).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(tmpl, text="Auto-apply on open (use selected)", variable=self.auto_apply_template,
                        command=self.toggle_auto_apply).pack(anchor="w", pady=(4,0))

        # Solver selection
        solver_box = ttk.Labelframe(right, text="Fitting method"); solver_box.pack(fill=tk.X, pady=4)
        for key in ["classic", "modern_vp", "modern_trf", "lmfit_vp"]:
            rb = ttk.Radiobutton(
                solver_box,
                text=SOLVER_LABELS.get(key, key),
                variable=self.solver_choice,
                value=key,
                command=self._on_solver_change,
            )
            if key == "lmfit_vp" and not self.has_lmfit:
                rb.state(["disabled"])
                add_tooltip(rb, "Install lmfit to enable.")
            rb.pack(anchor="w")

        opts_parent = ttk.Frame(solver_box)
        opts_parent.pack(fill=tk.X, pady=2)
        self.solver_frames = {}

        # Classic options
        f_classic = ttk.Frame(opts_parent)
        rowc = ttk.Frame(f_classic)
        rowc.pack(anchor="w")
        ttk.Label(rowc, text="Max evals").pack(side=tk.LEFT)
        ttk.Entry(rowc, width=7, textvariable=self.classic_maxfev).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(f_classic, text="Centers in window", variable=self.classic_centers_window).pack(anchor="w")

        def _toggle_classic_adv():
            if self.classic_adv_frame.winfo_ismapped():
                self.classic_adv_frame.pack_forget()
            else:
                self.classic_adv_frame.pack(anchor="w")

        ttk.Button(f_classic, text="Advanced (Classic)", command=_toggle_classic_adv).pack(anchor="w", pady=(4, 0))
        self.classic_adv_frame = ttk.Frame(f_classic)

        rowm = ttk.Frame(self.classic_adv_frame)
        rowm.pack(anchor="w")
        ttk.Label(rowm, text="Window margin frac").pack(side=tk.LEFT)
        ttk.Entry(rowm, width=4, textvariable=self.classic_margin).pack(side=tk.LEFT, padx=2)
        row1 = ttk.Frame(self.classic_adv_frame)
        row1.pack(anchor="w")
        ttk.Label(row1, text="Min FWHM×Δx").pack(side=tk.LEFT)
        ttk.Entry(row1, width=4, textvariable=self.classic_fwhm_min).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(self.classic_adv_frame)
        row2.pack(anchor="w")
        ttk.Label(row2, text="Max span frac").pack(side=tk.LEFT)
        ttk.Entry(row2, width=4, textvariable=self.classic_fwhm_max).pack(side=tk.LEFT, padx=2)
        row3 = ttk.Frame(self.classic_adv_frame)
        row3.pack(anchor="w")
        ttk.Label(row3, text="Max height ×").pack(side=tk.LEFT)
        ttk.Entry(row3, width=4, textvariable=self.classic_height_factor).pack(side=tk.LEFT, padx=2)

        self.solver_frames["classic"] = f_classic

        # Modern options (shared for VP and TRF)
        f_modern = ttk.Frame(opts_parent)
        r1 = ttk.Frame(f_modern); r1.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(r1, text="Loss").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=self.modern_loss, state="readonly",
                     values=MODERN_LOSSES, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(r1, text="Weights").pack(side=tk.LEFT, padx=(6,0))
        ttk.Combobox(r1, textvariable=self.modern_weight, state="readonly",
                     values=MODERN_WEIGHTS, width=8).pack(side=tk.LEFT, padx=2)
        r2 = ttk.Frame(f_modern); r2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(r2, text="f_scale").pack(side=tk.LEFT)
        ttk.Entry(r2, width=6, textvariable=self.modern_fscale).pack(side=tk.LEFT, padx=2)
        ttk.Label(r2, text="Max evals").pack(side=tk.LEFT, padx=(6,0))
        ttk.Entry(r2, width=7, textvariable=self.modern_maxfev).pack(side=tk.LEFT, padx=2)
        ttk.Label(r2, text="Restarts").pack(side=tk.LEFT, padx=(6,0))
        ttk.Entry(r2, width=4, textvariable=self.modern_restarts).pack(side=tk.LEFT, padx=2)
        ttk.Label(r2, text="Jitter %").pack(side=tk.LEFT, padx=(6,0))
        ttk.Entry(r2, width=4, textvariable=self.modern_jitter).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(f_modern, text="Centers in window", variable=self.modern_centers_window).pack(anchor="w")
        ttk.Checkbutton(f_modern, text="Min FWHM ≈2×Δx", variable=self.modern_min_fwhm).pack(anchor="w")
        self.solver_frames["modern_vp"] = f_modern
        self.solver_frames["modern_trf"] = f_modern

        # LMFIT options
        f_lmfit = ttk.Frame(opts_parent)
        ttk.Label(f_lmfit, text="Algo").pack(side=tk.LEFT)
        ttk.Combobox(f_lmfit, textvariable=self.lmfit_algo, state="readonly",
                     values=LMFIT_ALGOS, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_lmfit, text="Max evals").pack(side=tk.LEFT, padx=(6,0))
        ttk.Entry(f_lmfit, width=7, textvariable=self.lmfit_maxfev).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(f_lmfit, text="Share FWHM", variable=self.lmfit_share_fwhm).pack(anchor="w", padx=(4,0))
        ttk.Checkbutton(f_lmfit, text="Share η", variable=self.lmfit_share_eta).pack(anchor="w", padx=(4,0))
        self.solver_frames["lmfit_vp"] = f_lmfit

        self._show_solver_opts()

        # Uncertainty panel
        unc_box = ttk.Labelframe(right, text="Uncertainty"); unc_box.pack(fill=tk.X, pady=4)
        unc_method_frame = ttk.Labelframe(unc_box, text="Method")
        unc_method_frame.pack(fill=tk.X, padx=4, pady=4)
        self._unc_method_frame = unc_method_frame
        boot_solver_label = SOLVER_LABELS.get(self._select_bootstrap_solver(), self._select_bootstrap_solver())
        boot_label = f"Bootstrap (base solver = {boot_solver_label})"
        self._unc_asym_radio = ttk.Radiobutton(
            unc_method_frame,
            text="Asymptotic",
            value="Asymptotic",
            variable=self.unc_method,
            command=self._on_unc_method_change,
        )
        self._unc_asym_radio.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._unc_boot_radio = ttk.Radiobutton(
            unc_method_frame,
            text=boot_label,
            value=boot_label,
            variable=self.unc_method,
            command=self._on_unc_method_change,
        )
        self._unc_boot_radio.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self._unc_bayes_radio = ttk.Radiobutton(
            unc_method_frame,
            text="Bayesian",
            value="Bayesian",
            variable=self.unc_method,
            command=self._on_unc_method_change,
        )
        self._unc_bayes_radio.grid(row=0, column=2, sticky="w", padx=4, pady=2)

        # Bootstrap advanced options (visible only when method == "bootstrap")
        self._boot_options_row = ttk.Frame(unc_box)
        ttk.Label(self._boot_options_row, text="Draws:").pack(side=tk.LEFT)
        self.boot_n_var = tk.IntVar(value=int(self.cfg.get("bootstrap_n", 500)))
        ttk.Spinbox(
            self._boot_options_row,
            from_=50,
            to=100000,
            increment=50,
            width=8,
            textvariable=self.boot_n_var,
            command=lambda: self._cfg_set("bootstrap_n", int(self.boot_n_var.get())),
        ).pack(side=tk.LEFT, padx=4)

        action_row = ttk.Frame(unc_box)
        action_row.pack(fill=tk.X, pady=(2, 2))
        self._unc_action_row = action_row
        ttk.Button(action_row, text="Run", command=self.run_uncertainty).pack(side=tk.LEFT, padx=4)
        self.chk_ci_band = ttk.Checkbutton(
            action_row,
            text="Show uncertainty band",
            variable=self.show_ci_band_var,
        )
        self.chk_ci_band.pack(side=tk.LEFT, padx=4)
        self.ci_toggle = self.chk_ci_band
        self.band_chk = self.chk_ci_band
        self._ci_toggle_chk = self.chk_ci_band
        self._set_ci_toggle_state(False)
        # Guard so forcing off for Bayesian won't persist/override user's saved pref
        self._suspend_ci_trace = False

        def _ci_trace_guard(*_e):
            # Guard against re-entrancy from var.set()
            if getattr(self, "_suspend_ci_trace", False):
                return
            try:
                # Persist per-method preference on any user change
                try:
                    mk = self._unc_selected_method_key()
                    if mk:
                        self._band_pref[mk] = bool(self.show_ci_band_var.get())
                        self._persist_band_prefs()
                except Exception:
                    pass
                # Toggle rendering to match the new state (no auto-unchecking)
                self._toggle_ci_band()
            except Exception:
                return

        try:
            for info in list(self.show_ci_band.trace_info() or []):
                # Tk may return ('write', cb) or ('w', cb) and sometimes include more items
                mode = info[0] if len(info) > 0 else None
                cbname = info[1] if len(info) > 1 else None
                if mode in ("write", "w") and cbname:
                    # trace_remove expects the exact mode token it gave you
                    self.show_ci_band.trace_remove(
                        mode if mode in ("write", "read", "unset", "w", "r", "u") else "write",
                        cbname,
                    )
        except Exception:
            pass
        self.show_ci_band.trace_add("write", _ci_trace_guard)
        self._ensure_compute_toggle()

        # Extra uncertainty controls
        unc_frame = ttk.Frame(unc_box)
        unc_frame.pack(fill=tk.X, pady=2)
        r = 0
        # --- Bootstrap controls (hidden unless Bootstrap is selected) ---
        self._jitter_label = ttk.Label(unc_frame, text="Jitter %")
        self._jitter_label.grid(row=r, column=0, sticky="e")
        self._jitter_entry = ttk.Entry(unc_frame, textvariable=self.bootstrap_jitter_var, width=6)
        self._jitter_entry.grid(row=r, column=1, sticky="w")
        # Clamp jitter% on blur to [0, 50]
        try:
            self._jitter_entry.bind("<FocusOut>", lambda _e: self._safe_jitter_pct())
        except Exception:
            pass
        r += 1
        # Bootstrap refit solver override
        self._boot_solver_label = ttk.Label(unc_frame, text="Refit solver")
        self._boot_solver_label.grid(row=r, column=0, sticky="e")
        boot_solver = self.cfg.get("unc_boot_solver", self.solver_choice.get())
        if not getattr(self, "has_lmfit", False) and str(boot_solver).lower().startswith("lmfit"):
            boot_solver = self.solver_choice.get()
            self.cfg["unc_boot_solver"] = boot_solver
            save_config(self.cfg)
        self.boot_solver_choice = tk.StringVar(value=str(boot_solver))
        if self.boot_solver_choice.get() and self.boot_solver_choice.get() != self.solver_choice.get():
            self.boot_solver_overridden = True
        # Only offer LMFIT if actually available
        def _available_solvers():
            ordered = ["classic", "modern_vp", "modern_trf"]
            if getattr(self, "has_lmfit", False):
                ordered.append("lmfit_vp")
            return tuple(ordered)
        self._boot_solver_cb = ttk.Combobox(
            unc_frame,
            textvariable=self.boot_solver_choice,
            width=22,
            state="readonly",
            values=_available_solvers(),
        )
        self._boot_solver_cb.grid(row=r, column=1, columnspan=2, sticky="w")
        # If persisted override isn't available here, fall back to base solver
        try:
            _vals = self._as_seq(self._boot_solver_cb.cget("values"))
            if self.boot_solver_choice.get() not in _vals:
                self.boot_solver_choice.set(self.solver_choice.get())
                self._cfg_set("unc_boot_solver", self.boot_solver_choice.get())
                self.boot_solver_overridden = False
        except Exception:
            pass
        self._boot_solver_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_boot_solver_choice(),
        )
        r += 1
        # ------------------------------------------------------------------
        # Bootstrap residual resampling controls (encapsulated in a frame;
        # shown only when Bootstrap is the selected uncertainty method)
        # ------------------------------------------------------------------
        # These controls pass through cfg values as-is; None stays None (no stringification).
        self._boot_opts_frame = ttk.LabelFrame(unc_frame, text="Bootstrap options")
        self._boot_opts_frame.grid(row=r, column=0, columnspan=4, sticky="we", pady=(4, 2))
        r += 1

        br = 0  # row inside _boot_opts_frame
        ttk.Label(self._boot_opts_frame, text="Resampling mode:").grid(row=br, column=0, sticky="e")
        _resampling_init = self.cfg.get("boot_resampling")
        self.boot_resampling_var = tk.StringVar(
            value=(str(_resampling_init).lower() if _resampling_init is not None else "")
        )
        self.boot_resampling_menu = ttk.Combobox(
            self._boot_opts_frame,
            textvariable=self.boot_resampling_var,
            state="readonly",
            values=("classic", "wild"),
            width=12,
        )
        self.boot_resampling_menu.grid(row=br, column=1, sticky="w", padx=2)
        self.boot_resampling_menu.bind(
            "<<ComboboxSelected>>",
            lambda _e: [self._cfg_set("boot_resampling", self.boot_resampling_var.get()),
                        self._update_bootstrap_modes_state()],
        )
        br += 1

        ttk.Label(self._boot_opts_frame, text="Wild weights:").grid(row=br, column=0, sticky="e")
        _wild_init = self.cfg.get("boot_wild_weights")
        self.boot_wild_weights_var = tk.StringVar(
            value=(str(_wild_init).lower() if _wild_init is not None else "")
        )
        self.boot_wild_weights_menu = ttk.Combobox(
            self._boot_opts_frame,
            textvariable=self.boot_wild_weights_var,
            state="disabled",  # enabled only when mode == 'wild'
            values=("rademacher", "mammen", "gaussian"),
            width=12,
        )
        self.boot_wild_weights_menu.grid(row=br, column=1, sticky="w", padx=2)
        self.boot_wild_weights_menu.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._cfg_set("boot_wild_weights", self.boot_wild_weights_var.get()),
        )
        br += 1

        # Studentized residuals by default, as batch (PR #252: +0.6 to 2.2 coverage points)
        self.boot_studentize_var = tk.BooleanVar(
            value=bool(self.cfg.get("boot_studentize", True))
        )
        self.boot_studentize_chk = ttk.Checkbutton(
            self._boot_opts_frame,
            text="Studentize residuals",
            variable=self.boot_studentize_var,
            command=lambda: self._cfg_set("boot_studentize", bool(self.boot_studentize_var.get())),
        )
        self.boot_studentize_chk.grid(row=br, column=0, columnspan=2, sticky="w", padx=2)
        br += 1

        # Clamp controls live inside the same frame
        clamp_label = ttk.Label(self._boot_opts_frame, text="Center clamp mode:")
        clamp_label.grid(row=br, column=0, sticky="e")
        _mode_init = self.cfg.get("boot_center_clamp_mode")
        self.boot_center_clamp_mode = tk.StringVar(
            value=(str(_mode_init) if _mode_init is not None else "none")
        )
        clamp_mode = ttk.Combobox(
            self._boot_opts_frame,
            textvariable=self.boot_center_clamp_mode,
            state="readonly",
            values=("none", "abs", "frac_fwhm", "frac_bounds"),
            width=12,
        )
        clamp_mode.grid(row=br, column=1, sticky="w", padx=2)
        clamp_mode.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._cfg_set("boot_center_clamp_mode", self.boot_center_clamp_mode.get()),
        )
        br += 1
        clamp_val_label = ttk.Label(self._boot_opts_frame, text="Clamp value:")
        clamp_val_label.grid(row=br, column=0, sticky="e")
        _val_init = self.cfg.get("boot_center_clamp_value")
        self.boot_center_clamp_value = tk.StringVar(
            value=("" if _val_init is None else str(_val_init))
        )
        clamp_val_entry = ttk.Entry(self._boot_opts_frame, textvariable=self.boot_center_clamp_value, width=10)
        clamp_val_entry.grid(row=br, column=1, sticky="w", padx=2)
        clamp_val_entry.bind(
            "<FocusOut>",
            lambda _e: self._cfg_set(
                "boot_center_clamp_value", _to_float_or_none(self.boot_center_clamp_value.get())
            ),
        )
        # Set initial enable/disable and initial visibility
        self._update_bootstrap_modes_state()
        self._update_bootstrap_visibility()

        # continue placing generic (method-agnostic) controls below
        ttk.Label(unc_frame, text="CI α").grid(row=r, column=0, sticky="e")
        _alpha_entry = ttk.Entry(unc_frame, textvariable=self.alpha_var, width=6)
        _alpha_entry.grid(row=r, column=1, sticky="w")

        try:
            _alpha_entry.bind("<FocusOut>", lambda _e: self._safe_alpha())
        except Exception:
            pass

        ttk.Checkbutton(unc_frame, text="Center residuals", variable=self.center_resid_var).grid(row=r, column=2, sticky="w")
        r += 1

        # --- Bayesian controls (hidden unless Bayesian is selected) ---
        self._bayes_frame = ttk.LabelFrame(unc_frame, text="Bayesian")
        self._bayes_frame.grid(row=r, column=0, columnspan=4, sticky="we", pady=(4, 0))
        ttk.Label(self._bayes_frame, text="Walkers").grid(row=0, column=0, sticky="e")
        walkers_entry = ttk.Entry(self._bayes_frame, textvariable=self.bayes_walkers_var, width=6)
        walkers_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(self._bayes_frame, text="Burn-in").grid(row=0, column=2, sticky="e")
        burn_entry = ttk.Entry(self._bayes_frame, textvariable=self.bayes_burn_var, width=6)
        burn_entry.grid(row=0, column=3, sticky="w")
        ttk.Label(self._bayes_frame, text="Steps").grid(row=1, column=0, sticky="e")
        steps_entry = ttk.Entry(self._bayes_frame, textvariable=self.bayes_steps_var, width=6)
        steps_entry.grid(row=1, column=1, sticky="w")
        ttk.Label(self._bayes_frame, text="Thin").grid(row=1, column=2, sticky="e")
        thin_entry = ttk.Entry(self._bayes_frame, textvariable=self.bayes_thin_var, width=6)
        thin_entry.grid(row=1, column=3, sticky="w")
        ttk.Label(self._bayes_frame, text="σ prior").grid(row=2, column=0, sticky="e")
        ttk.Combobox(
            self._bayes_frame,
            textvariable=self.bayes_prior_var,
            width=12,
            values=("half_cauchy", "half_normal"),
            state="readonly",
        ).grid(row=2, column=1, sticky="w")
        try:
            walkers_entry.bind("<FocusOut>", lambda _e: self._safe_bayes_walkers())
            burn_entry.bind("<FocusOut>", lambda _e: self._safe_bayes_burn())
            steps_entry.bind("<FocusOut>", lambda _e: self._safe_bayes_steps())
            thin_entry.bind("<FocusOut>", lambda _e: self._safe_bayes_thin())
        except Exception:
            pass
        r += 1

        bayes_band_frame = ttk.Labelframe(unc_frame, text="Bayesian band")
        bayes_band_frame.grid(row=r, column=0, columnspan=4, sticky="we", pady=(4, 0))
        self._bayes_band_frame = bayes_band_frame
        ttk.Checkbutton(
            bayes_band_frame,
            text="Enable posterior band",
            variable=self.bayes_band_enabled_var,
            command=lambda: self._cfg_set("bayes_band_enabled", bool(self.bayes_band_enabled_var.get())),
        ).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(
            bayes_band_frame,
            text="Force band (ignore diagnostics)",
            variable=self.bayes_band_force_var,
            command=lambda: self._cfg_set("bayes_band_force", bool(self.bayes_band_force_var.get())),
        ).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(bayes_band_frame, text="Max draws (default 512):").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(bayes_band_frame, width=8, textvariable=self.bayes_band_max_draws_var).grid(row=1, column=1, sticky="w")
        self.bayes_band_max_draws_var.trace_add(
            "write",
            lambda *_: self._on_bayes_band_max_draws_change(),
        )

        diag_frame = ttk.Labelframe(bayes_band_frame, text="Diagnostics gates")
        diag_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=4, pady=4)
        ttk.Checkbutton(
            diag_frame,
            text="Enable diagnostics (ESS/R̂/MCSE)",
            variable=self.bayes_diag_var,
            command=lambda: self._cfg_set("bayes_diagnostics", bool(self.bayes_diag_var.get())),
        ).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(diag_frame, text="ESS min (default 200):").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(diag_frame, width=8, textvariable=self.bayes_thresh_ess_min_var).grid(row=1, column=1, sticky="w")
        self.bayes_thresh_ess_min_var.trace_add(
            "write",
            lambda *_: self._on_bayes_diag_ess_change(),
        )
        ttk.Label(diag_frame, text="R̂ max (default 1.05):").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Entry(diag_frame, width=8, textvariable=self.bayes_thresh_rhat_max_var).grid(row=2, column=1, sticky="w")
        self.bayes_thresh_rhat_max_var.trace_add(
            "write",
            lambda *_: self._on_bayes_diag_rhat_change(),
        )
        ttk.Label(diag_frame, text="MCSE mean (blank = ∞):").grid(row=3, column=0, sticky="w", padx=4)
        ttk.Entry(diag_frame, width=8, textvariable=self.bayes_thresh_mcse_mean_str).grid(row=3, column=1, sticky="w")
        self.bayes_thresh_mcse_mean_str.trace_add(
            "write",
            lambda *_: self._on_bayes_diag_mcse_change(),
        )
        r += 1

        # --- Bootstrap ties (visible + enabled only when Bootstrap + LMFIT refit solver) ---
        self._boot_ties_frame = ttk.LabelFrame(unc_frame, text="Bootstrap ties (LMFIT only)")
        self._boot_ties_frame.grid(row=r, column=0, columnspan=4, sticky="we", pady=(4, 0))
        self.chk_share_fwhm = ttk.Checkbutton(
            self._boot_ties_frame, text="Share FWHM across peaks", variable=self.lmfit_share_fwhm
        )
        self.chk_share_fwhm.grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.chk_share_eta = ttk.Checkbutton(
            self._boot_ties_frame, text="Share η across peaks", variable=self.lmfit_share_eta
        )
        self.chk_share_eta.grid(row=0, column=1, sticky="w", padx=2, pady=1)

        self._update_unc_widgets()
        self._on_uncertainty_method_changed()

        # Axes / label controls
        axes_box = ttk.Labelframe(right, text="Axes / Labels")
        axes_box.pack(fill=tk.X, pady=6)
        row1 = ttk.Frame(axes_box); row1.pack(fill=tk.X)
        ttk.Label(row1, text="X-axis label:").pack(side=tk.LEFT)
        self.x_label_entry = ttk.Entry(row1, width=16, textvariable=self.x_label_var)
        self.x_label_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Apply", command=self.apply_x_label).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(axes_box); row2.pack(fill=tk.X, pady=(2,0))
        ttk.Button(row2, text="Superscript", command=self.insert_superscript).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Subscript", command=self.insert_subscript).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(row2, text="Auto-format", variable=self.x_label_auto_math,
                        command=self._on_x_label_auto_math_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Save as default", command=self.save_x_label_default).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(row2, text="Show legend", variable=self.show_legend_var,
                        command=self._on_legend_toggle).pack(side=tk.LEFT, padx=2)

        # Performance panel
        perf_box = ttk.Labelframe(right, text="Performance"); perf_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(perf_box, text="Numba", variable=self.perf_numba).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Cache baseline", variable=self.perf_cache_baseline).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Seed all", variable=self.perf_seed_all).pack(anchor="w")
        par = ttk.Labelframe(perf_box, text="Parallelism")
        par.pack(fill=tk.X, pady=4)
        ttk.Radiobutton(
            par,
            text="Outer: many workers (BLAS=1)",
            value="outer",
            variable=self.perf_parallel_strategy,
            command=lambda: self.request_apply_performance(source="radio:strategy_outer", immediate=True, save_cfg=True),
        ).pack(anchor="w")
        row = ttk.Frame(par)
        row.pack(fill=tk.X)
        ttk.Radiobutton(
            row,
            text="Inner: BLAS threads per task =",
            value="inner",
            variable=self.perf_parallel_strategy,
            command=self.apply_performance,
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            row,
            from_=0,
            to=256,
            width=4,
            textvariable=self.perf_blas_threads,
            command=self.apply_performance,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(row, text="(0 = library default)").pack(side=tk.LEFT, padx=(6, 0))
        rowp = ttk.Frame(perf_box); rowp.pack(fill=tk.X, pady=2)
        ttk.Label(rowp, text="Seed:").pack(side=tk.LEFT)
        ttk.Entry(rowp, width=8, textvariable=self.seed_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(rowp, text="Unc. workers (0=auto):").pack(side=tk.LEFT, padx=(8,0))
        ttk.Spinbox(
            rowp,
            from_=0,
            to=256,
            textvariable=self.perf_unc_workers,
            width=6,
            command=self.apply_performance,
        ).pack(side=tk.LEFT)
        ttk.Button(rowp, text="Apply", command=self._apply_performance_immediate).pack(side=tk.LEFT, padx=4)

        # Batch processing
        batch_box = ttk.Labelframe(right, text="Batch"); batch_box.pack(fill=tk.X, pady=4)
        rowb1 = ttk.Frame(batch_box); rowb1.pack(fill=tk.X, pady=2)
        ttk.Label(rowb1, text="Pattern").pack(side=tk.LEFT)
        ttk.Entry(rowb1, width=22, textvariable=self.batch_patterns).pack(side=tk.LEFT, padx=4)
        rowb2 = ttk.Frame(batch_box); rowb2.pack(fill=tk.X, pady=2)
        ttk.Label(rowb2, text="Source:").pack(side=tk.LEFT)
        ttk.Radiobutton(rowb2, text="Current", variable=self.batch_source, value="current").pack(side=tk.LEFT)
        ttk.Radiobutton(rowb2, text="Template", variable=self.batch_source, value="template").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(rowb2, text="Auto", variable=self.batch_source, value="auto").pack(side=tk.LEFT)
        rowb3 = ttk.Frame(batch_box); rowb3.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(rowb3, text="Re-height", variable=self.batch_reheight).pack(side=tk.LEFT)
        ttk.Checkbutton(rowb3, text="Save traces", variable=self.batch_save_traces).pack(side=tk.LEFT, padx=4)
        ttk.Label(rowb3, text="Auto max:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Spinbox(rowb3, from_=1, to=20, textvariable=self.batch_auto_max, width=5).pack(side=tk.LEFT)
        rowb4 = ttk.Frame(batch_box); rowb4.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(rowb4, text="Compute uncertainty in batch", variable=self.batch_unc_enabled).pack(side=tk.LEFT)
        ttk.Button(batch_box, text="Run Batch…", command=self.run_batch).pack(side=tk.LEFT, pady=4)

        # Status bar and log
        bar = ttk.Frame(self.root); bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Open CSV/TXT/DAT, set baseline/range, add peaks, set η, then Fit.")
        self.status = ttk.Label(bar, textvariable=self.status_var)
        self.status.pack(side=tk.LEFT, padx=6)
        self.log_btn = ttk.Button(bar, text="Show log \u25B8", command=self.toggle_log)
        self.log_btn.pack(side=tk.RIGHT)
        self.abort_btn = tk.Button(
            bar,
            text="Abort",
            command=self.on_abort_clicked,
            bg="#b91c1c",
            fg="white",
            activebackground="#991b1b",
            relief="raised",
            bd=1,
        )
        self.abort_btn.config(state=tk.DISABLED)
        self.abort_btn.pack(side=tk.RIGHT, padx=(6, 6))
        self.pbar = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pbar.pack(side=tk.RIGHT, padx=6)
        self.progress = self.pbar
        self._log_console = None
        self._log_visible = False
        self._log_frame = None
        if not hasattr(self, "_log_buffer"):
            self._log_buffer = []
        if not hasattr(self, "_auto_show_log_on_warn"):
            self._auto_show_log_on_warn = True
        if not hasattr(self, "_auto_show_log_on_error"):
            self._auto_show_log_on_error = True

        # Initial peak list height
        self.refresh_tree()

    def _show_solver_opts(self):
        for f in self.solver_frames.values():
            f.pack_forget()
        frame = self.solver_frames.get(self.solver_choice.get())
        if frame:
            frame.pack(side=tk.LEFT, padx=4)

    def _solver_options(self, choice: Optional[str] = None) -> dict:
        solver = choice or self.solver_choice.get()
        # Per-peak width caps (None where disabled/locked)
        width_caps = [
            (None if (p.lock_width or not bool(getattr(p, "fwhm_cap_enabled", False)))
             else float(getattr(p, "fwhm_cap_value", float("inf"))))
            for p in self.peaks
        ]
        if solver in ("modern_vp", "modern_trf"):
            min_fwhm = 1e-6
            if self.modern_min_fwhm.get() and self.x is not None and self.x.size > 1:
                min_fwhm = 2.0 * float(np.median(np.diff(self.x)))
            return {
                "loss": self.modern_loss.get(),
                "weights": self.modern_weight.get(),
                "f_scale": float(self.modern_fscale.get()),
                "maxfev": int(self.modern_maxfev.get()),
                "restarts": int(self.modern_restarts.get()),
                "jitter_pct": float(self.modern_jitter.get()),
                "centers_in_window": bool(self.modern_centers_window.get()),
                "min_fwhm": float(min_fwhm),
                "width_caps": width_caps,
            }
        if solver == "lmfit_vp":
            return {
                "algo": self.lmfit_algo.get(),
                "maxfev": int(self.lmfit_maxfev.get()),
                "share_fwhm": bool(self.lmfit_share_fwhm.get()),
                "share_eta": bool(self.lmfit_share_eta.get()),
                "width_caps": width_caps,
            }
        return {
            "maxfev": int(self.classic_maxfev.get()),
            "bound_centers_to_window": bool(self.classic_centers_window.get()),
            "margin_frac": float(self.classic_margin.get()),
            "fwhm_min_factor": float(self.classic_fwhm_min.get()),
            "fwhm_max_factor": float(self.classic_fwhm_max.get()),
            "height_factor": float(self.classic_height_factor.get()),
            "width_caps": width_caps,
        }

    def _settings(self) -> dict:
        """Every setting the core path reads, from the widgets (Tk thread only).

        One dict for Fit, Export and Run Batch (game plan 2.1), so that a fit, its
        exported table and a batch over the same files use the same values. Batch
        adds its own defaults for the keys the GUI leaves out: noise_model and
        baseline_fit (decisions D1 and D2).
        """

        solver = self.solver_choice.get()
        opts = self._solver_options(solver)
        self._safe_jitter_pct()  # clamps the jitter field and keeps the user's value
        boot = self._bootstrap_cfg_from_gui()
        try:
            unc_workers = int(float(self.perf_unc_workers.get()))
        except Exception:
            unc_workers = int(self.cfg.get("perf_unc_workers", 0) or 0)
        settings = {
            "solver": solver,
            solver: opts,
            # the Modern solvers' restarts, which run_fit_consistent runs itself
            "solver_restarts": int(opts.get("restarts", 1) or 1),
            "solver_jitter_pct": float(opts.get("jitter_pct", 0.0) or 0.0),
            "mode": self.baseline_mode.get(),
            "use_baseline": bool(self.use_baseline.get()),
            "baseline": self._current_baseline_cfg(),
            "baseline_uses_fit_range": bool(self.baseline_use_range.get()),
            "perf_numba": bool(self.perf_numba.get()),
            "perf_cache_baseline": bool(self.perf_cache_baseline.get()),
            "perf_seed_all": bool(self.perf_seed_all.get()),
            "perf_seed": boot["seed"] if boot["seed"] is not None else "",
            "perf_unc_workers": unc_workers,
            "perf_parallel_strategy": str(self.perf_parallel_strategy.get()),
            "perf_blas_threads": int(self.perf_blas_threads.get()),
            "unc_method": self._unc_selected_method_key(),
            "unc_alpha": float(self._safe_alpha()),
            "unc_center_resid": bool(self.center_resid_var.get()),
            "unc_boot_solver": boot["solver"],
            "bootstrap_n": int(boot["n"]),
            "bootstrap_jitter": float(boot["jitter"]),
            "boot_studentize": bool(self.boot_studentize_var.get()),
            # whether a band is computed: the GUI's Compute band, not its Show band
            "ui_band_pref_bootstrap": bool(self._should_compute_ci_band()),
            "export_unc_wide": bool(self.cfg.get("export_unc_wide", False)),
            "lmfit_share_fwhm": bool(self.lmfit_share_fwhm.get()),
            "lmfit_share_eta": bool(self.lmfit_share_eta.get()),
            "bayes_band_enabled": bool(self.bayes_band_enabled_var.get()),
            "bayes_band_force": bool(self.bayes_band_force_var.get()),
            "bayes_band_max_draws": self._bayes_band_max_draws_value(),
            "bayes_diagnostics": bool(self.bayes_diag_var.get()),
            "bayes_diag_ess_min": self._bayes_diag_ess_min_value(),
            "bayes_diag_rhat_max": self._bayes_diag_rhat_max_value(),
            "bayes_diag_mcse_mean": self._bayes_diag_mcse_value(),
            "bayes_walkers": int(self._safe_bayes_walkers()),
            "bayes_burn": int(self._safe_bayes_burn()),
            "bayes_steps": int(self._safe_bayes_steps()),
            "bayes_thin": int(self._safe_bayes_thin()),
            "bayes_prior_sigma": str(self.bayes_prior_var.get() or self.cfg.get("bayes_prior_sigma", "half_cauchy")),
            "log_debug_mode": bool(self.debug_mode_var.get()),
        }
        # the bootstrap's other knobs, as the user set them (absent: the method's default)
        for key in ("boot_resampling", "boot_wild_weights", "boot_center_clamp_mode", "boot_center_clamp_value"):
            if self.cfg.get(key) not in (None, ""):
                settings[key] = self.cfg[key]
        # refits with another solver take that solver's own options
        if boot["solver"] != solver and boot["solver"] in SOLVER_LABELS:
            settings[boot["solver"]] = self._solver_options(boot["solver"])
        if self.fit_xmin is not None and self.fit_xmax is not None:
            settings["fit_xmin"] = float(self.fit_xmin)
            settings["fit_xmax"] = float(self.fit_xmax)
        # keep what the user set for the next start
        for key in ("bootstrap_n", "unc_center_resid", "bayes_walkers", "bayes_burn", "bayes_steps", "bayes_thin",
                    "bayes_diagnostics"):
            self.cfg[key] = settings[key]
        save_config(self.cfg)
        return settings

    def _fit_inputs(self):
        """What a fit of the current state sees: x, y, the baseline it holds fixed
        (zeros with Use baseline off), the fit window's mask and the mode."""

        base = self.baseline if (self.use_baseline.get() and self.baseline is not None) else np.zeros_like(self.x)
        return self.x, self.y_raw, np.asarray(base, float), self.current_fit_mask(), self.baseline_mode.get()

    def _peak_state(self) -> list:
        return [(p.center, p.height, p.fwhm, p.eta, bool(p.lock_center), bool(p.lock_width)) for p in self.peaks]

    def _current_fit(self) -> Optional[dict]:
        """The last fit, while it is still the fit of what the app shows: the same
        data, baseline, window and mode, and the peaks as the fit left them."""

        rec = getattr(self, "_fit", None)
        if rec is None or self.x is None or self.y_raw is None:
            return None
        x, y, base, mask, mode = self._fit_inputs()
        same = (
            mode == rec["mode"]
            and self._peak_state() == rec["peaks"]
            and all(np.array_equal(a, b) for a, b in ((x, rec["x"]), (y, rec["y"]), (base, rec["baseline"]), (mask, rec["mask"])))
        )
        return rec if same else None

    def _export_fit(self) -> dict:
        """The fit an export describes: the current one, or the peaks as they are,
        which no fit produced (fit_ok False)."""

        rec = self._current_fit()
        if rec is not None:
            return rec["result"]
        x, y, base, mask, mode = self._fit_inputs()
        return pipeline.unfitted(x, y, self.peaks, base, mode, mask)

    @log_action
    def _on_boot_solver_choice(self, *_):
        if hasattr(self, "boot_solver_choice"):
            chosen = self.boot_solver_choice.get()
        else:
            chosen = None
        if chosen is not None:
            try:
                self._cfg_set("unc_boot_solver", chosen)
            except Exception:
                try:
                    self.cfg["unc_boot_solver"] = chosen
                except Exception:
                    pass
        self.boot_solver_overridden = True
        if chosen is not None:
            try:
                self.dlog(
                    f"Uncertainty solver manually set to '{chosen}' (overriding main solver selection)"
                )
            except Exception:
                pass
        try:
            self._update_unc_widgets()
            self._update_bootstrap_tie_widgets()
        except Exception:
            pass

    @log_action
    def _on_solver_change(self, *_):
        new_solver = self.solver_choice.get()
        self.solver_title.set(SOLVER_LABELS.get(new_solver, new_solver))
        self.cfg["solver_choice"] = new_solver
        save_config(self.cfg)
        if not getattr(self, "boot_solver_overridden", False):
            self.boot_solver_overridden = False
            try:
                if hasattr(self, "boot_solver_choice"):
                    self.boot_solver_choice.set(new_solver)
            except Exception:
                pass
            try:
                self._cfg_set("unc_boot_solver", new_solver)
            except Exception:
                try:
                    self.cfg["unc_boot_solver"] = new_solver
                except Exception:
                    pass
            try:
                self.dlog(
                    f"Syncing bootstrap solver to main solver '{new_solver}' by default"
                )
            except Exception:
                pass
        self._show_solver_opts()
        self._update_unc_widgets()
        try:
            self._update_bootstrap_tie_widgets()
        except Exception:
            pass

    @log_action
    def _update_unc_widgets(self):
        # Keep the Bootstrap label in sync with the actual solver selection
        chosen = self._select_bootstrap_solver()
        label = SOLVER_LABELS.get(chosen, chosen)
        boot_label = f"Bootstrap (base solver = {label})"
        current = self.unc_method.get()
        if hasattr(self, "_unc_boot_radio"):
            try:
                self._unc_boot_radio.configure(text=boot_label, value=boot_label)
            except Exception:
                pass
        if current.startswith("Bootstrap"):
            self.unc_method.set(boot_label)
        self._on_uncertainty_method_changed()
        self._debug_snapshot("_update_unc_widgets")

    @log_action
    def _on_unc_method_change(self, _e=None):
        label = self.unc_method.get()
        self._dbg(f"_on_unc_method_change: label='{label}'")
        self._debug_snapshot("_on_unc_method_change(pre)")
        method_key = method_display_to_key(label)
        self.cfg["unc_method"] = method_key
        try:
            unc_section = self.cfg.setdefault("uncertainty", {})
            unc_section["method"] = method_key
            unc_section.setdefault("return_band", bool(self.show_ci_band_var.get()))
            if "compute_band" not in unc_section:
                unc_section["compute_band"] = self._should_compute_ci_band()
        except Exception:
            pass
        save_config(self.cfg)
        self._update_unc_widgets()
        self._on_uncertainty_method_changed()
        self._debug_snapshot("_on_unc_method_change(post)")

    def _as_seq(self, v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        if isinstance(v, str):
            return tuple(s for s in v.split() if s)
        try:
            return tuple(v)
        except TypeError:
            return tuple()

    def _safe_alpha(self):
        try:
            raw = self.alpha_var.get()
        except Exception:
            raw = None
        a = normalize_float(raw, 0.05)
        a = min(0.49, max(1e-6, a))
        self.alpha_var.set(a)
        try:
            self._cfg_set("unc_alpha", a)
        except Exception:
            pass
        return a

    def _safe_int_from_var(self, var: tk.Variable, default: int) -> int:
        try:
            raw = var.get()
        except Exception:
            raw = default
        try:
            return int(raw)
        except Exception:
            try:
                return int(float(str(raw).strip()))
            except Exception:
                return default

    def _safe_float_from_var(self, var: tk.Variable, default: float) -> float:
        try:
            raw = var.get()
        except Exception:
            raw = default
        try:
            return float(raw)
        except Exception:
            try:
                return float(str(raw).strip())
            except Exception:
                return default

    def _safe_bayes_walkers(self) -> int:
        try:
            raw = self.bayes_walkers_var.get()
        except Exception:
            raw = None
        v = max(0, normalize_int(raw, 0))
        # enforce minimal requirement if user sets a too-small positive number
        try:
            ndim = 4 * len(self.peaks)
            min_walkers = max(2 * max(1, ndim), 4)
            if v and v < min_walkers:
                v = min_walkers
        except Exception:
            pass
        try:
            self.bayes_walkers_var.set(v)
        except Exception:
            pass
        return v

    def _safe_bayes_burn(self) -> int:
        try:
            raw = self.bayes_burn_var.get()
        except Exception:
            raw = None
        v = max(0, normalize_int(raw, 0))
        try:
            self.bayes_burn_var.set(v)
        except Exception:
            pass
        return v

    def _safe_bayes_steps(self) -> int:
        try:
            raw = self.bayes_steps_var.get()
        except Exception:
            raw = None
        v = normalize_int(raw, 0)
        if v < 0:
            v = 0
        try:
            self.bayes_steps_var.set(v)
        except Exception:
            pass
        return v

    def _safe_bayes_thin(self) -> int:
        try:
            raw = self.bayes_thin_var.get()
        except Exception:
            raw = None
        v = max(1, normalize_int(raw, 1))
        try:
            self.bayes_thin_var.set(v)
        except Exception:
            pass
        return v

    def _bayes_band_max_draws_value(self) -> int:
        val = self._safe_int_from_var(self.bayes_band_max_draws_var, 512)
        return val if val > 0 else 512

    def _bayes_diag_ess_min_value(self) -> float:
        val = self._safe_float_from_var(self.bayes_thresh_ess_min_var, 200.0)
        return val if val > 0 else 200.0

    def _bayes_diag_rhat_max_value(self) -> float:
        val = self._safe_float_from_var(self.bayes_thresh_rhat_max_var, 1.05)
        return val if val > 0 else 1.05

    def _bayes_diag_mcse_value(self) -> float:
        try:
            txt = str(self.bayes_thresh_mcse_mean_str.get()).strip()
        except Exception:
            txt = ""
        if not txt:
            return float("inf")
        try:
            val = float(txt)
        except Exception:
            return float("inf")
        return val if val > 0 else float("inf")

    @log_action
    def _on_bayes_band_max_draws_change(self):
        self._cfg_set("bayes_band_max_draws", self._bayes_band_max_draws_value())

    @log_action
    def _on_bayes_diag_ess_change(self):
        self._cfg_set("bayes_diag_ess_min", self._bayes_diag_ess_min_value())

    @log_action
    def _on_bayes_diag_rhat_change(self):
        self._cfg_set("bayes_diag_rhat_max", self._bayes_diag_rhat_max_value())

    @log_action
    def _on_bayes_diag_mcse_change(self):
        self._cfg_set("bayes_diag_mcse_mean", self._bayes_diag_mcse_value())

    def _safe_jitter_pct(self):
        try:
            raw = self.bootstrap_jitter_var.get()
        except Exception:
            raw = None
        v = normalize_float(raw, 0.0)
        v = min(50.0, max(0.0, v))
        self.bootstrap_jitter_var.set(v)
        try:
            self._cfg_set("bootstrap_jitter", v / 100.0)
        except Exception:
            pass
        return v



    def _suspend_clicks(self):
        """Disable click-to-add regardless of checkbox state."""
        try:
            if getattr(self, "cid", None) is not None:
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None
        except Exception:
            self.cid = None

    def _restore_clicks(self):
        """Restore click-to-add to match the current checkbox state."""
        try:
            if self.cid is None and self.add_peaks_mode.get():
                self.cid = self.canvas.mpl_connect("button_press_event", self.on_click_plot)
        except Exception:
            self.cid = None

    def _on_add_peaks_toggle(self):
        self.cfg["ui_add_peaks_on_click"] = bool(self.add_peaks_mode.get())
        save_config(self.cfg)
        if self.add_peaks_mode.get():
            self._restore_clicks()
        else:
            self._suspend_clicks()

    def _toggle_ci_band(self, *_):
        val = bool(self.show_ci_band_var.get())
        self._dbg(f"_toggle_ci_band: value={val}")
        self.cfg["ui_show_uncertainty_band"] = val
        try:
            unc_section = self.cfg.setdefault("uncertainty", {})
            unc_section["return_band"] = val
        except Exception:
            pass
        save_config(self.cfg)
        self.refresh_plot()
        self._debug_snapshot("_toggle_ci_band", show_band=val)

    def _should_compute_ci_band(self) -> bool:
        try:
            var = getattr(self, "compute_ci_band_var", None)
            if var is not None:
                return bool(var.get())
        except Exception:
            pass
        try:
            return bool(self.show_ci_band_var.get())
        except Exception:
            return True

    def _set_ci_toggle_state(self, enabled: bool):
        try:
            widget = getattr(self, "band_chk", getattr(self, "ci_toggle", None))
            if widget is not None:
                widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        except Exception:
            pass
        try:
            state = ("!disabled",) if enabled else ("disabled",)
            self.ci_toggle.state(state)
        except Exception:
            pass
        try:
            chk = getattr(self, "_compute_chk", None)
            if chk is not None:
                chk.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass

    # --- Uncertainty helpers (UI-side) ------------------------------------------
    def _band_tuple_from(self, m):
        """
        Return (x, lo, hi) or None.
        Accepts:
          - m['band'] or m['prediction_band'] as a 3-tuple
          - or separate m['band_x'], m['band_lo'], m['band_hi']
        """
        import numpy as _np
        band = m.get("band") or m.get("prediction_band")
        if band is None:
            bx, blo, bhi = m.get("band_x"), m.get("band_lo"), m.get("band_hi")
            if bx is not None and blo is not None and bhi is not None:
                band = (bx, blo, bhi)
        if band is None:
            return None
        if isinstance(band, (list, tuple)) and len(band) == 3:
            x, lo, hi = [_np.asarray(v) for v in band]
            if x.size and lo.size and hi.size and x.shape == lo.shape == hi.shape:
                return (x, lo, hi)
        return None

    @log_action
    def _on_uncertainty_method_changed(self, *_):
        if getattr(self, "_unc_change_busy", False):
            return
        self._unc_change_busy = True
        try:
            method_key = self._unc_selected_method_key()

            # Save the old method's band/compute prefs before switching
            try:
                old_key = getattr(self, "_last_unc_method_key", None)
                if old_key:
                    if old_key in self._band_pref:
                        self._band_pref[old_key] = bool(self.show_ci_band_var.get())
                    if self.compute_ci_band_var is not None:
                        self._compute_pref[old_key] = bool(self.compute_ci_band_var.get())
                self._persist_band_prefs()
            except Exception:
                pass

            self._ensure_compute_toggle()
            try:
                self._suspend_ci_trace = True
                default_compute = method_key != "bayesian"
                if self.compute_ci_band_var is not None:
                    self.compute_ci_band_var.set(
                        bool(self._compute_pref.get(method_key, default_compute))
                    )
                self.show_ci_band_var.set(
                    bool(self._band_pref.get(method_key, False))
                )
            finally:
                self._suspend_ci_trace = False

            self._set_ci_toggle_state(True)
            if self._compute_chk is not None:
                try:
                    self._compute_chk.state(["!disabled"])
                except Exception:
                    pass

            is_boot = method_key == "bootstrap"
            is_bayes = method_key == "bayesian"
            # Show only the controls for the selected method
            try:
                (self._jitter_label.grid if is_boot else self._jitter_label.grid_remove)()
                (self._jitter_entry.grid if is_boot else self._jitter_entry.grid_remove)()
            except Exception:
                pass
            try:
                (self._boot_solver_label.grid if is_boot else self._boot_solver_label.grid_remove)()
                (self._boot_solver_cb.grid if is_boot else self._boot_solver_cb.grid_remove)()
            except Exception:
                pass
            try:
                (self._boot_ties_frame.grid if is_boot else self._boot_ties_frame.grid_remove)()
            except Exception:
                pass
            self._update_bootstrap_visibility()
            try:
                (self._bayes_frame.grid if is_bayes else self._bayes_frame.grid_remove)()
            except Exception:
                pass
            bayes_band = getattr(self, "_bayes_band_frame", None)
            if bayes_band is not None:
                try:
                    (bayes_band.grid if is_bayes else bayes_band.grid_remove)()
                except Exception:
                    pass
            # keep bootstrap jitter enabled/disabled appropriately when visible
            try:
                self._jitter_entry.configure(state=("normal" if is_boot else "disabled"))
            except Exception:
                pass
            # Track current method for next switch
            self._last_unc_method_key = method_key
            boot_cb = getattr(self, "_boot_solver_cb", None)
            if boot_cb is not None:
                try:
                    boot_cb.configure(state="readonly")
                except Exception:
                    pass
            options_row = getattr(self, "_boot_options_row", None)
            if options_row is not None:
                if is_boot:
                    try:
                        options_row.pack_forget()
                        options_row.pack(fill=tk.X, pady=2, before=self._unc_action_row)
                    except Exception:
                        pass
                else:
                    try:
                        options_row.pack_forget()
                    except Exception:
                        pass

            # refresh Bootstrap tie widgets (enable only for Bootstrap + LMFIT refit solver)
            self._update_bootstrap_tie_widgets()
            # for asymptotic/bootstrap the checkbox remains user-controlled
            try:
                unc_section = self.cfg.setdefault("uncertainty", {})
                unc_section["method"] = method_key
                unc_section["return_band"] = bool(self.show_ci_band_var.get())
                unc_section["compute_band"] = self._should_compute_ci_band()
                self.cfg["unc_method"] = method_key
                save_config(self.cfg)
            except Exception:
                pass
            self._debug_snapshot("_on_uncertainty_method_changed", method_key=method_key)
        finally:
            self._unc_change_busy = False


    def _toggle_ci_compute(self, *_):
        try:
            var = getattr(self, "compute_ci_band_var", None)
            if var is None:
                return
            val = bool(var.get())
            key = self._unc_selected_method_key()
            if key:
                self._compute_pref[key] = val
                self._persist_compute_prefs()
        except Exception:
            pass
        self._update_bootstrap_visibility()


    def _update_bootstrap_modes_state(self):
        """Enable/disable wild-weights control based on selected resampling mode."""
        try:
            mode = (self.boot_resampling_var.get() or "").strip().lower()
        except Exception:
            mode = ""
        state = ("readonly" if mode == "wild" else "disabled")
        try:
            self.boot_wild_weights_menu.configure(state=state)
        except Exception:
            pass

        self._update_bootstrap_visibility()


    def _update_bootstrap_tie_widgets(self):
        try:
            sel = str(self.unc_method_var.get()).lower()
            is_boot = ("bootstrap" in sel)
            solver = self._select_bootstrap_solver()
            is_lmfit = str(solver).lower().startswith("lmfit") and bool(getattr(self, "has_lmfit", False))
            state = "normal" if (is_boot and is_lmfit) else "disabled"
            for w in (
                getattr(self, "chk_share_fwhm", None),
                getattr(self, "chk_share_eta", None),
            ):
                if w is not None:
                    w.configure(state=state)
        except Exception:
            pass

    def on_abort_clicked(self):
        try:
            for evt in list(self._jobs):
                evt.set()
            self._progress_end()
            self.status_warn("Aborting…")
        except Exception:
            pass

    # --- Jobs the Abort button stops: Run, Export and Run Batch ------------------
    def _job_begin(self) -> threading.Event:
        """A new job's own abort event. Abort sets every running job's; the button
        stays enabled while any job runs, so a job that ends never disarms another."""

        evt = threading.Event()
        self._jobs.add(evt)
        self.abort_btn.config(state=tk.NORMAL)
        return evt

    def _job_end(self, evt: threading.Event) -> None:
        self._jobs.discard(evt)
        if self._jobs:
            return
        try:
            self.abort_btn.config(state=tk.DISABLED)
        except tk.TclError:
            pass
        if self._closing:
            self.root.after(0, self.root.destroy)

    def _ending(self, evt: threading.Event, done):
        """``done`` for run_in_thread, then the end of the job ``evt``, whatever ``done`` does."""

        def finish(result, error):
            try:
                done(result, error)
            finally:
                self._job_end(evt)

        return finish

    def _on_close(self):
        """Close the window, after the running jobs stop.

        A job's worker thread dies with the app: an Export closed mid-write left
        half its files. So the jobs are aborted first, and the window goes when
        the last one has finished (an aborted Export writes its fit and trace).
        A second close does not wait."""

        if not self._jobs or self._closing:
            self.root.destroy()
            return
        self._closing = True
        self.on_abort_clicked()
        self.status_warn("Closing once the running job has stopped…")

    def _toggle_legend_action(self):
        self.show_legend_var.set(not self.show_legend_var.get())
        self._on_legend_toggle()

    def _on_eta_change(self):
        self.cfg["ui_eta"] = float(self.global_eta.get())
        save_config(self.cfg)

    def toggle_log(self):
        if self._log_visible:
            if self._log_console is not None and self._log_frame is not None:
                self._log_frame.pack_forget()
            self.log_btn.config(text="Show log \u25B8")
            self._log_visible = False
        else:
            self._ensure_log_panel_visible()

    def _ensure_log_panel_visible(self):
        if self._log_console is None:
            self._log_frame = ttk.Frame(self.root)
            btns = ttk.Frame(self._log_frame)
            ttk.Button(btns, text="Copy log", command=self._copy_log).pack(side=tk.LEFT)
            ttk.Button(btns, text="Save log…", command=self._save_log).pack(side=tk.LEFT)
            ttk.Checkbutton(
                btns,
                text="Debug",
                variable=self.debug_mode_var,
                command=self._on_debug_toggle,
            ).pack(side=tk.LEFT, padx=(12, 2))
            btns.pack(fill=tk.X)
            self._log_console = scrolledtext.ScrolledText(self._log_frame, height=8, state="disabled")
            self._log_console.configure(
                bg="#000000", fg="#00ff66", insertbackground="#00ff66", font=self.default_font
            )
            self._log_console.pack(fill=tk.BOTH, expand=True)
        self._log_console.configure(state="normal")
        self._log_console.delete("1.0", "end")
        if self._log_buffer:
            self._log_console.insert("end", "\n".join(self._log_buffer) + "\n")
        self._log_console.configure(state="disabled")
        self._log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_btn.config(text="Hide log \u25BE")
        self._log_visible = True

    def copy_log(self):  # back-compat alias
        return self._copy_log()

    def save_log(self):  # back-compat alias
        return self._save_log()

    def _copy_log(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(self._log_buffer))
        except Exception:
            pass

    def _save_log(self):
        path = filedialog.asksaveasfilename(title="Save log", defaultextension=".txt")
        if path:
            try:
                Path(path).write_text("\n".join(self._log_buffer), encoding="utf-8")
            except Exception as e:
                messagebox.showwarning("Log", f"Could not save log: {e}")

    def set_busy(self, flag: bool, msg: str = ""):
        self.status_var.set(msg or ("Working..." if flag else "Ready."))
        if flag:
            self.pbar.start(12)
        else:
            self.pbar.stop()
        self.root.update_idletasks()

    def log(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        glyph = {"INFO": "\u2139", "WARN": "\u26A0", "ERROR": "\u2716"}.get(level.upper(), "\u2139")
        line = f"[{ts}] {glyph} {level.upper()} \u2014 {msg}"
        with self._log_lock:
            self._log_buffer.append(line)
            if len(self._log_buffer) > LOG_MAX_LINES:
                del self._log_buffer[: len(self._log_buffer) - LOG_MAX_LINES]

        def ui_update():
            if hasattr(self, "status_var"):
                self.status_var.set(line[:120])
            if self._log_console is not None:
                try:
                    self._log_console.configure(state="normal")
                    self._log_console.insert("end", line + "\n")
                    self._log_console.see("end")
                    self._log_console.configure(state="disabled")
                except Exception:
                    pass
            level_u = level.upper()
            if (level_u == "WARN" and self._auto_show_log_on_warn) or (
                level_u == "ERROR" and self._auto_show_log_on_error
            ):
                self._ensure_log_panel_visible()

        if not self._on_tk_thread():
            self._ui_calls.put(ui_update)  # a worker: the Tk thread shows it when it polls
            return
        try:
            self.root.after(0, ui_update)
        except Exception:
            ui_update()

    def log_threadsafe(self, msg: str, level: str = "INFO"):
        self.log(msg, level)

    # --- Worker threads never call Tk (game plan 3.4) -------------------------
    POLL_MS = 25

    def _on_tk_thread(self) -> bool:
        return threading.current_thread() is self._tk_thread

    def call_on_main(self, fn) -> None:
        """Run ``fn`` on the Tk thread: now when called there, else when it polls."""

        if self._on_tk_thread():
            fn()
        else:
            self._ui_calls.put(fn)

    def _drain_ui_calls(self) -> None:
        while True:
            try:
                fn = self._ui_calls.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                traceback.print_exc()

    def log_info(self, msg: str) -> None:
        self.log(msg, "INFO")

    def log_warn(self, msg: str) -> None:
        self.log(msg, "WARN")

    def log_error(self, msg: str) -> None:
        self.log(msg, "ERROR")

    # --- Status + progress helpers (UI-safe, no-ops if widgets missing) ---
    def status_info(self, msg: str) -> None:
        try:
            if hasattr(self, "status") and self.status:
                self.status.config(text=msg)
            if hasattr(self, "status_var") and self.status_var:
                self.status_var.set(msg)
        except Exception:
            pass
        cb = getattr(self, "log_info", None)
        if callable(cb):
            try:
                cb(msg)
                return
            except Exception:
                pass
        print(msg)

    def status_warn(self, msg: str) -> None:
        try:
            if hasattr(self, "status") and self.status:
                self.status.config(text=msg)
            if hasattr(self, "status_var") and self.status_var:
                self.status_var.set(msg)
        except Exception:
            pass
        cb = getattr(self, "log_warn", None)
        if callable(cb):
            try:
                cb(msg)
                return
            except Exception:
                pass
        print("WARN:", msg)

    def status_error(self, msg: str) -> None:
        try:
            if hasattr(self, "status") and self.status:
                self.status.config(text=msg)
            if hasattr(self, "status_var") and self.status_var:
                self.status_var.set(msg)
        except Exception:
            pass
        cb = getattr(self, "log_error", None)
        if callable(cb):
            try:
                cb(msg)
                return
            except Exception:
                pass
        print("ERROR:", msg)

    def _progress_begin(self, tag: str = "task") -> None:
        # Start progressbar if present; tolerate absence.
        try:
            if getattr(self, "_progress_depth", None) is None:
                self._progress_depth = 0
            self._progress_depth += 1
            pb = getattr(self, "progress", None)
            if pb and hasattr(pb, "start"):
                # Use a small interval; Tk handles animation.
                pb.start(10)
        except Exception:
            pass

    def _progress_end(self, tag: str = "task") -> None:
        # Stop progressbar if present; tolerate absence.
        try:
            if getattr(self, "_progress_depth", None) is None:
                self._progress_depth = 0
            self._progress_depth = max(0, self._progress_depth - 1)
            if self._progress_depth == 0:
                pb = getattr(self, "progress", None)
                if pb and hasattr(pb, "stop"):
                    pb.stop()
        except Exception:
            pass

    def run_in_thread(self, fn, on_done):
        """Run ``fn()`` on a worker thread and ``on_done(result, error)`` on the Tk thread.

        The worker never calls Tk, which serves other threads only inside its main
        loop: the Tk thread polls for it with root.after and meanwhile runs what the
        worker queued (log lines, progress) through call_on_main.
        """

        box = {}

        def worker():
            try:
                box["result"] = fn()
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll():
            self._drain_ui_calls()
            if thread.is_alive():
                try:
                    self.root.after(self.POLL_MS, poll)
                except tk.TclError:
                    pass  # the window is gone
                return
            self._drain_ui_calls()
            on_done(box.get("result"), box.get("error"))

        self.root.after(self.POLL_MS, poll)

    def _new_figure(self):
        self.ax.clear()
        self.ax.set_xlabel(format_axis_label_inline(self.x_label_var.get(), self.x_label_auto_math.get()))
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Open a data file to begin")
        self.canvas.draw_idle()

    # ----- Zoom helpers -----
    def zoom_out(self, factor: float = 1.5):
        if self.x is None:
            return
        xmin_full, xmax_full = float(self.x.min()), float(self.x.max())
        cur_lo, cur_hi = self.ax.get_xlim()
        cur_lo = max(min(cur_lo, cur_hi), xmin_full)
        cur_hi = min(max(cur_lo, cur_hi), xmax_full)
        width = cur_hi - cur_lo
        if width <= 0:
            width = xmax_full - xmin_full
        new_width = min((xmax_full - xmin_full), width * factor)
        center = 0.5 * (cur_lo + cur_hi)
        new_lo = max(xmin_full, center - new_width / 2.0)
        new_hi = min(xmax_full, center + new_width / 2.0)
        if (new_hi - new_lo) < new_width:
            if new_lo <= xmin_full + 1e-12:
                new_hi = xmin_full + new_width
            elif new_hi >= xmax_full - 1e-12:
                new_lo = xmax_full - new_width
            new_lo = max(xmin_full, new_lo)
            new_hi = min(xmax_full, new_hi)
        self.ax.set_xlim(new_lo, new_hi)
        try:
            self.ax.relim(); self.ax.autoscale_view(scalex=False, scaley=True)
        except Exception:
            pass
        self.canvas.draw_idle()

    def reset_view(self):
        try:
            self.nav.home()
        except Exception:
            if self.x is not None:
                self.ax.set_xlim(float(self.x.min()), float(self.x.max()))
                self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _update_file_label(self, path: Optional[str]) -> None:
        if path:
            p = Path(path)
            self.current_file = p
            full = str(p)
            disp = full if len(full) <= 60 else "..." + full[-57:]
            self.file_label.config(text=disp)
            self.file_label.unbind("<Enter>")
            self.file_label.unbind("<Leave>")
            add_tooltip(self.file_label, full)
        else:
            self.current_file = None
            self.file_label.config(text="No file loaded")
            self.file_label.unbind("<Enter>")
            self.file_label.unbind("<Leave>")

    # ----- File handling -----
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Select spectrum data",
            filetypes=[
                ("Data files","*.csv *.txt *.dat"),
                ("CSV files","*.csv"),
                ("Text files","*.txt *.dat"),
                ("All files","*.*"),
            ]
        )
        if not path:
            return
        try:
            x, y = load_xy_any(path)
        except ValueError as e:
            self.status_warn(str(e))
            try:
                messagebox.showwarning(
                    "Unsupported File",
                    f"{e}\n\nTip: Use a two-column (x y) text file."
                )
            except Exception:
                pass
            return
        except Exception as e:
            messagebox.showerror("Open data", f"Failed to read file:\n{e}")
            return

        self.x, self.y_raw = x, y
        self._update_file_label(path)
        self.compute_baseline()
        self.peaks.clear()

        # Auto-apply selected template if enabled
        if self.auto_apply_template.get():
            t = self._templates()
            name = self.auto_apply_template_name.get()
            if name and name in t:
                self._apply_template_list(t[name], reheight=True)

        self.refresh_tree()
        self.refresh_plot()
        self._update_template_info()
        self.status_var.set("Loaded. Adjust baseline, (optionally) set fit range, add peaks; Fit.")

    # ----- Baseline -----
    def on_baseline_use_range_toggle(self, *_):
        self.cfg["baseline_uses_fit_range"] = bool(self.baseline_use_range.get())
        save_config(self.cfg)
        if self.y_raw is None:
            return
        self.compute_baseline()

    def _update_baseline_frames(self):
        method = self.base_method_var.get().strip().lower()
        if method == "als":
            self.poly_frame.forget()
            self.als_frame.pack(fill=tk.X, pady=2)
        else:
            self.als_frame.forget()
            self.poly_frame.pack(fill=tk.X, pady=2)

    def _on_base_method_change(self, *_):
        self._update_baseline_frames()
        if self.y_raw is not None:
            self.compute_baseline()

    def _range_mask(self):
        if self.x is None:
            return None
        if self.fit_xmin is None or self.fit_xmax is None:
            return None
        lo, hi = sorted((self.fit_xmin, self.fit_xmax))
        return (self.x >= lo) & (self.x <= hi)

    def compute_baseline(self):
        if self.y_raw is None:
            return
        method = self.base_method_var.get().strip().lower()
        use_slice = bool(self.baseline_use_range.get())
        mask = self._range_mask() if use_slice else None

        if method == "als":
            lam = float(self.als_lam.get())
            asym = float(self.als_asym.get())
            niter = int(self.als_niter.get())
            thresh = float(self.als_thresh.get())
            key = None
            if performance.cache_baseline_enabled():
                mkey = None
                if mask is not None and np.any(mask):
                    mkey = (float(self.x[mask][0]), float(self.x[mask][-1]))
                key = (hash(self.y_raw.tobytes()), lam, asym, niter, thresh, mkey)
            if key is not None and key in self._baseline_cache:
                self.baseline = self._baseline_cache[key]
            else:
                try:
                    if mask is None or not np.any(mask):
                        base = signals.als_baseline(
                            self.y_raw, lam=lam, p=asym, niter=niter, tol=thresh
                        )
                    else:
                        x_sub = self.x[mask]
                        y_sub = self.y_raw[mask]
                        z_sub = signals.als_baseline(
                            y_sub, lam=lam, p=asym, niter=niter, tol=thresh
                        )
                        base = np.interp(
                            self.x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1]
                        )
                    self.baseline = base
                    if key is not None:
                        self._baseline_cache[key] = base
                except Exception as e:
                    messagebox.showwarning("Baseline", f"ALS baseline failed: {e}")
                    self.baseline = np.zeros_like(self.y_raw)
        elif method == "polynomial":
            deg = int(self.poly_degree_var.get())
            norm_x = bool(self.poly_norm_var.get())
            key = None
            if performance.cache_baseline_enabled():
                slice_key = None
                if mask is not None and np.any(mask):
                    slice_key = (float(self.x[mask][0]), float(self.x[mask][-1]))
                key = (hash(self.y_raw.tobytes()), "poly", int(deg), bool(norm_x), slice_key)
            if key is not None and key in self._baseline_cache:
                self.baseline = self._baseline_cache[key]
            else:
                try:
                    base, used_deg = signals.polynomial_baseline(
                        self.x,
                        self.y_raw,
                        degree=deg,
                        mask=mask,
                        normalize_x=norm_x,
                        return_used_degree=True,
                    )
                    if used_deg != deg:
                        self.poly_degree_var.set(used_deg)
                        self.status_var.set(
                            f"Polynomial baseline: degree reduced to {used_deg} due to fit-range points."
                        )
                    self.baseline = base
                    if key is not None:
                        self._baseline_cache[key] = base
                except Exception as e:
                    messagebox.showwarning("Baseline", f"Polynomial baseline failed: {e}")
                    self.baseline = np.zeros_like(self.y_raw)
        else:
            messagebox.showwarning("Baseline", f"Unknown baseline method: {method}")
            self.baseline = np.zeros_like(self.y_raw)

        try:
            y_t = self.get_fit_target()
            snr = signals.snr_estimate(y_t if y_t is not None else self.y_raw)
            self.snr_text.set(f"S/N: {snr:.2f}")
        except Exception:
            self.snr_text.set("S/N: --")

        self.refresh_plot()

    def save_baseline_default(self):
        self.cfg["baseline_defaults"] = {
            "method": self.base_method_var.get().strip().lower(),
            "als": {
                "lam": float(self.als_lam.get()),
                "p": float(self.als_asym.get()),
                "niter": int(self.als_niter.get()),
                "thresh": float(self.als_thresh.get()),
            },
            "polynomial": {
                "degree": int(self.poly_degree_var.get()),
                "normalize_x": bool(self.poly_norm_var.get()),
            },
        }
        save_config(self.cfg)
        messagebox.showinfo("Baseline", "Saved as default for future sessions.")

    def _current_baseline_cfg(self):
        base = {"method": self.base_method_var.get().strip().lower()}
        if base["method"] == "polynomial":
            base.update(
                {
                    "degree": int(self.poly_degree_var.get()),
                    "normalize_x": bool(self.poly_norm_var.get()),
                }
            )
        elif base["method"] == "als":
            base.update(
                {
                    "lam": float(self.als_lam.get()),
                    "p": float(self.als_asym.get()),
                    "niter": int(self.als_niter.get()),
                    "thresh": float(self.als_thresh.get()),
                }
            )
        return base

    # ----- Signals for seeding and fitting -----
    def get_seed_signal(self):
        if self.y_raw is None:
            return None
        if self.use_baseline.get() and self.baseline is not None:
            return self.y_raw - self.baseline
        return self.y_raw

    def get_fit_target(self):
        if self.y_raw is None:
            return None
        if self.use_baseline.get() and self.baseline is not None and self.baseline_mode.get() == "subtract":
            return self.y_raw - self.baseline
        return self.y_raw

    def current_fit_mask(self) -> Optional[np.ndarray]:
        if self.x is None:
            return None
        if self.fit_xmin is None or self.fit_xmax is None:
            return np.ones_like(self.x, dtype=bool)
        lo, hi = sorted((self.fit_xmin, self.fit_xmax))
        return (self.x >= lo) & (self.x <= hi)

    # ----- Fit range helpers -----
    def enable_span(self):
        if self._span_active and self._span is not None:
            return

        self._span_prev_click_toggle = bool(self.add_peaks_mode.get())
        self.add_peaks_checkbox.configure(state="disabled")
        self.add_peaks_mode.set(False)
        self._suspend_clicks()
        self._span_active = True
        self.status_var.set("Drag to select fit range… ESC to cancel")
        self._span_cids = []
        try:
            widget = self.canvas.get_tk_widget()
            self._cursor_before_span = widget.cget("cursor") or ""
            widget.configure(cursor="tcross")
        except Exception:
            self._cursor_before_span = ""

        def _finish(_ok: bool):
            try:
                if self._span is not None:
                    try:
                        self._span.set_active(False)
                    except Exception:
                        pass
                    try:
                        self._span.disconnect_events()
                    except Exception:
                        pass
            except Exception:
                pass
            self._span = None
            for cid in self._span_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._span_cids = []
            try:
                self.add_peaks_checkbox.configure(state="normal")
            except Exception:
                pass
            self.add_peaks_mode.set(self._span_prev_click_toggle)
            try:
                self.canvas.get_tk_widget().configure(cursor=self._cursor_before_span)
            except Exception:
                pass
            self._restore_clicks()
            self._span_active = False

        def onselect(xmin, xmax):
            try:
                self.fit_xmin, self.fit_xmax = float(xmin), float(xmax)
                self.fit_min_var.set(f"{self.fit_xmin:.6g}")
                self.fit_max_var.set(f"{self.fit_xmax:.6g}")
                if self.baseline_use_range.get():
                    self.compute_baseline()
                else:
                    self.refresh_plot()
                self.status_var.set("Range selected.")
            finally:
                _finish(True)

        def on_key(event):
            if event.key == "escape" and self._span_active:
                _finish(False)
                self.status_var.set("Range selection canceled.")

        def on_leave(_event):
            if self._span_active:
                _finish(False)
                self.status_var.set("Range selection canceled.")

        try:
            self._span = SpanSelector(
                self.ax,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.15, facecolor="tab:blue"),
            )
        except TypeError:
            self._span = SpanSelector(
                self.ax,
                onselect,
                "horizontal",
                useblit=True,
                rectprops=dict(alpha=0.15, facecolor="tab:blue"),
            )

        def on_close(_event):
            if self._span_active:
                _finish(False)
                self.status_var.set("Range selection canceled.")

        self._span_cids = [
            self.canvas.mpl_connect("key_press_event", on_key),
            self.canvas.mpl_connect("figure_leave_event", on_leave),
            self.canvas.mpl_connect("close_event", on_close),
        ]

    def apply_fit_range_from_fields(self):
        if self.x is None:
            return
        try:
            xmin = float(self.fit_min_var.get())
            xmax = float(self.fit_max_var.get())
        except Exception:
            messagebox.showwarning("Fit range", "Please enter numeric Min/Max.")
            return
        if xmin == xmax:
            messagebox.showwarning("Fit range", "Min and Max cannot be equal.")
            return
        self.fit_xmin, self.fit_xmax = xmin, xmax
        if self.baseline_use_range.get():
            self.compute_baseline()
        else:
            self.refresh_plot()

    def clear_fit_range(self):
        self.fit_xmin = self.fit_xmax = None
        self.fit_min_var.set("")
        self.fit_max_var.set("")
        if self.baseline_use_range.get():
            self.compute_baseline()
        else:
            self.refresh_plot()

    # ----- Axes label helpers -----
    def _on_x_label_auto_math_toggle(self):
        self.cfg["x_label_auto_math"] = bool(self.x_label_auto_math.get())
        save_config(self.cfg)
        self.apply_x_label()

    def insert_superscript(self):
        self.x_label_entry.insert(tk.INSERT, "$^{ }$")
        self.x_label_entry.icursor(self.x_label_entry.index(tk.INSERT) - 3)
        self.x_label_entry.focus_set()

    def insert_subscript(self):
        self.x_label_entry.insert(tk.INSERT, "$_{ }$")
        self.x_label_entry.icursor(self.x_label_entry.index(tk.INSERT) - 3)
        self.x_label_entry.focus_set()

    def apply_x_label(self):
        label = format_axis_label_inline(self.x_label_var.get(), self.x_label_auto_math.get())
        self.ax.set_xlabel(label)
        self.canvas.draw_idle()

    def save_x_label_default(self):
        self.cfg["x_label"] = self.x_label_var.get()
        save_config(self.cfg)
        messagebox.showinfo("Axes", f'Saved default x-axis label: "{self.x_label_var.get()}"')

    def _on_legend_toggle(self):
        self.cfg["ui_show_legend"] = bool(self.show_legend_var.get())
        save_config(self.cfg)
        self.refresh_plot()

    def _on_legend_sigfigs_change(self):
        try:
            val = int(self.legend_center_sigfigs.get())
        except Exception:
            val = 6
        self.cfg["legend_center_sigfigs"] = val
        save_config(self.cfg)
        self.refresh_plot()

    # ----- Templates helpers -----
    def _templates(self) -> dict:
        t = self.cfg.get("templates", {})
        if not isinstance(t, dict):
            t = {}
        return t

    def _persist_last_template(self, name: str) -> None:
        if name:
            self.cfg["last_template_name"] = name
            self.auto_apply_template_name.set(name)
            save_config(self.cfg)

    def _update_template_info(self):
        t = self._templates()
        names = sorted(t.keys())
        self.template_combo["values"] = names
        current = self.template_var.get()
        if current not in names:
            pref = self.auto_apply_template_name.get()
            if pref in names:
                current = pref
            elif names:
                current = names[0]
            else:
                current = ""
        self.template_var.set(current)
        self.template_info.config(text=f"Templates: {len(names)}")

    def save_template_as(self):
        name = simpledialog.askstring("Save template", "Template name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        t = self._templates()
        if name in t:
            if not messagebox.askyesno("Overwrite template", f"Template '{name}' exists. Overwrite?"):
                return
        t[name] = self.serialize_peaks()
        self.cfg["templates"] = t
        save_config(self.cfg)
        self.template_var.set(name)
        self.auto_apply_template_name.set(name)
        self.cfg["auto_apply_template_name"] = name
        save_config(self.cfg)
        self._update_template_info()
        messagebox.showinfo("Template", f"Saved {len(t[name])} peak(s) to template '{name}'.")

    def save_changes_to_selected_template(self):
        name = self.template_var.get()
        if not name:
            messagebox.showinfo("Template", "No template selected.")
            return
        t = self._templates()
        if name not in t:
            messagebox.showinfo("Template", f"Template '{name}' not found.")
            return
        if not messagebox.askyesno("Overwrite template",
                                   f"Overwrite template '{name}' with current peaks?"):
            return
        t[name] = self.serialize_peaks()
        self.cfg["templates"] = t
        save_config(self.cfg)
        self._update_template_info()
        messagebox.showinfo("Template", f"Saved changes to '{name}' ({len(t[name])} peak(s)).")

    def apply_selected_template(self):
        name = self.template_var.get()
        if not name:
            messagebox.showinfo("Template", "No template selected.")
            return
        t = self._templates()
        if name not in t:
            messagebox.showinfo("Template", f"Template '{name}' not found.")
            return
        if self.x is None or self.y_raw is None:
            messagebox.showinfo("Template", "Load a spectrum first (Open Data…).")
            return
        self._apply_template_list(t[name], reheight=True)
        self.status_var.set(f"Applied template '{name}' with {len(self.peaks)} peak(s).")
        self._persist_last_template(name)

    def delete_selected_template(self):
        name = self.template_var.get()
        if not name:
            messagebox.showinfo("Template", "No template selected.")
            return
        t = self._templates()
        if name not in t:
            messagebox.showinfo("Template", f"Template '{name}' not found.")
            return
        if not messagebox.askyesno("Delete template", f"Delete template '{name}'?"):
            return
        del t[name]
        self.cfg["templates"] = t
        if self.auto_apply_template_name.get() == name:
            self.auto_apply_template_name.set("")
            self.cfg["auto_apply_template_name"] = ""
        save_config(self.cfg)
        self._update_template_info()

    def toggle_auto_apply(self):
        self.cfg["auto_apply_template"] = bool(self.auto_apply_template.get())
        self.cfg["auto_apply_template_name"] = self.template_var.get()
        self.auto_apply_template_name.set(self.cfg["auto_apply_template_name"])
        save_config(self.cfg)

    # ----- Template application (data ops) -----
    def serialize_peaks(self) -> list:
        return [
            {"center": float(p.center), "height": float(p.height), "fwhm": float(p.fwhm),
             "eta": float(p.eta), "lock_width": bool(p.lock_width), "lock_center": bool(p.lock_center),
             "fwhm_cap_enabled": bool(getattr(p, "fwhm_cap_enabled", False)),
             "fwhm_cap_value": float(getattr(p, "fwhm_cap_value", 50.0))}
            for p in self.peaks
        ]

    def _apply_template_list(self, saved: list, reheight: bool = True):
        if self.x is None or self.y_raw is None or not saved:
            return
        sig = self.get_seed_signal()
        new = []
        for spk in saved:
            c = float(spk.get("center", 0.0))
            if c < self.x.min() or c > self.x.max():
                continue
            h = float(spk.get("height", 1.0))
            if reheight:
                y_at = float(np.interp(c, self.x, sig))
                h = max(y_at - float(np.median(sig)), 1e-6)
            new.append(Peak(center=c,
                            height=h,
                            fwhm=float(spk.get("fwhm", 5.0)),
                            eta=float(np.clip(spk.get("eta", 0.5), 0, 1)),
                            lock_width=bool(spk.get("lock_width", False)),
                            lock_center=bool(spk.get("lock_center", False)),
                            fwhm_cap_enabled=bool(spk.get("fwhm_cap_enabled", False)),
                            fwhm_cap_value=float(spk.get("fwhm_cap_value", 50.0))))
        new.sort(key=lambda p: p.center)
        self.peaks = new
        self.refresh_tree()
        self.refresh_plot()

    # ----- Peaks -----
    def on_click_plot(self, event):
        if self.x is None or event.inaxes != self.ax:
            return
        if self._span_active:
            return
        if getattr(event, "button", 1) != 1:
            return
        nav = getattr(self, "nav", None)
        mode = (getattr(nav, "mode", "") or getattr(nav, "_active", "") or "").upper()
        if "PAN" in mode or "ZOOM" in mode:
            return
        if not self.add_peaks_mode.get():
            return

        x0 = float(event.xdata)
        sig = self.get_seed_signal()
        y_at = float(np.interp(x0, self.x, sig))
        default_h = max(y_at - float(np.median(sig)), 1e-6)
        xr = float(self.x.max() - self.x.min())
        default_w = max(xr * 0.05, float(np.mean(np.diff(np.sort(self.x)))) * 5.0)
        pk = Peak(center=x0, height=default_h, fwhm=default_w, eta=float(self.global_eta.get()))
        self.peaks.append(pk)
        self.peaks.sort(key=lambda p: p.center)
        self.refresh_tree()
        self.refresh_plot()

    def auto_seed(self, max_peaks: int = 5):
        if self.x is None:
            return
        sig = self.get_seed_signal()
        mask = self.current_fit_mask()
        sig2 = sig.copy()
        if mask is not None:
            sig2[~mask] = np.min(sig2)
        prom = 0.1 * (sig2.max() - sig2.min())
        idx, props = find_peaks(sig2, prominence=prom)
        if len(idx) == 0:
            messagebox.showinfo("Auto-seed", "No peaks found in the selected range. Adjust baseline/range.")
            return
        order = np.argsort(props["prominences"])[::-1]
        idx = idx[order][:max_peaks]
        xr = float(self.x.max() - self.x.min())
        default_w = max(xr * 0.05, float(np.mean(np.diff(np.sort(self.x)))) * 5.0)
        self.peaks = []
        med = np.median(sig[mask]) if mask is not None else np.median(sig)
        for i in idx:
            h = max(sig[i] - med, 1e-6)
            self.peaks.append(Peak(center=float(self.x[i]), height=float(h),
                                   fwhm=float(default_w), eta=float(self.global_eta.get())))
        self.peaks.sort(key=lambda p: p.center)
        self.refresh_tree()
        self.fit()

    def apply_eta_all(self):
        eta = float(self.global_eta.get())
        for p in self.peaks:
            p.eta = float(np.clip(eta, 0, 1))
        self.refresh_plot()
        self._on_eta_change()

    def on_select_peak(self, _evt=None):
        sel = self._selected_index()
        if sel is None:
            return
        pk = self.peaks[sel]
        self.center_var.set(pk.center)
        self.height_var.set(pk.height)     # <-- fixed indentation
        self.fwhm_var.set(pk.fwhm)
        self.lockw_var.set(pk.lock_width)
        self.lockc_var.set(pk.lock_center)
        # Populate cap widgets
        try:
            self.fwhm_cap_var.set(float(getattr(pk, "fwhm_cap_value", max(1.0, pk.fwhm*5))))
            self.fwhm_cap_on.set(bool(getattr(pk, "fwhm_cap_enabled", False)))
        except Exception:
            self.fwhm_cap_var.set(50.0); self.fwhm_cap_on.set(False)
        # Disable editing when width is locked
        try:
            self.ent_fwhm_cap.configure(state=("disabled" if pk.lock_width else "normal"))
        except Exception:
            pass

    def on_lock_toggle(self):
        sel = self._selected_index()
        if sel is None:
            return
        pk = self.peaks[sel]
        pk.lock_width  = bool(self.lockw_var.get())
        pk.lock_center = bool(self.lockc_var.get())
        # Keep cap toggle/UI consistent with lock state
        if pk.lock_width:
            self.fwhm_cap_on.set(False)
        try:
            self.ent_fwhm_cap.configure(state=("disabled" if pk.lock_width else "normal"))
        except Exception:
            pass
        self.refresh_tree(keep_selection=True)
        self.refresh_plot()

    def on_cap_toggle(self):
        """Persist FWHM cap immediately. If 'All' is on, apply to every peak."""
        cap_on = bool(self.fwhm_cap_on.get())
        try:
            cap_val = float(self.fwhm_cap_var.get() or 50.0)
        except Exception:
            cap_val = 50.0
        if bool(self.fwhm_cap_all.get()):
            # Apply universally to all peaks (except those with locked width)
            for pk_all in self.peaks:
                if pk_all.lock_width:
                    continue
                pk_all.fwhm_cap_enabled = cap_on
                pk_all.fwhm_cap_value = cap_val
        else:
            # Apply only to the selected peak
            sel = self._selected_index()
            if sel is None:
                return
            pk = self.peaks[sel]
            if pk.lock_width:
                self.fwhm_cap_on.set(False)
                try:
                    self.ent_fwhm_cap.configure(state="disabled")
                except Exception:
                    pass
                return
            pk.fwhm_cap_enabled = cap_on
            pk.fwhm_cap_value = cap_val
        # Mirror and persist current options + UI prefs so uncertainty/batch and next session see them
        self._sync_cap_config()
        self.refresh_tree(keep_selection=True)
        self.refresh_plot()

    def _commit_cap_entry(self):
        """Enter key in the cap entry: if 'All' is on, propagate to all; else edit selected."""
        if bool(self.fwhm_cap_all.get()):
            self.on_cap_toggle()
        else:
            self.apply_edits()
        self._sync_cap_config()

    def _sync_cap_config(self):
        """Persist current cap UI prefs and options to the config file."""
        try:
            self.cfg["ui_fwhm_cap_all"] = bool(self.fwhm_cap_all.get())
            # Be tolerant to bad entry content
            try:
                self.cfg["ui_fwhm_cap_value"] = float(self.fwhm_cap_var.get() or 50.0)
            except Exception:
                self.cfg["ui_fwhm_cap_value"] = 50.0
            # Also mirror solver_options so downstream paths see updated width_caps
            opts = self._solver_options()
            self.cfg["solver_options"] = dict(opts)
            save_config(self.cfg)
        except Exception:
            pass

    def add_peak_from_fields(self):
        try:
            c = float(self.center_var.get())
            h = float(self.height_var.get())
            w = max(float(self.fwhm_var.get()), 1e-6)
        except ValueError:
            messagebox.showwarning("Add Peak", "Center, Height, and FWHM must be numbers.")
            return
        pk = Peak(center=c, height=h, fwhm=w, eta=float(self.global_eta.get()),
                  lock_center=bool(self.lockc_var.get()), lock_width=bool(self.lockw_var.get()),
                  fwhm_cap_enabled=bool(self.fwhm_cap_on.get()),
                  fwhm_cap_value=float(self.fwhm_cap_var.get() or 50.0))
        self.peaks.append(pk)
        self.peaks.sort(key=lambda p: p.center)
        self.refresh_tree()
        self.refresh_plot()

    def apply_edits(self):
        sel = self._selected_index()
        if sel is None:
            return
        pk = self.peaks[sel]
        try:
            pk.center = float(self.center_var.get())
            pk.height = float(self.height_var.get())
            pk.fwhm   = max(float(self.fwhm_var.get()), 1e-6)
            pk.lock_width  = bool(self.lockw_var.get())
            pk.lock_center = bool(self.lockc_var.get())
            if not pk.lock_width:
                pk.fwhm_cap_enabled = bool(self.fwhm_cap_on.get())
                pk.fwhm_cap_value   = float(self.fwhm_cap_var.get() or 50.0)
        except ValueError:
            messagebox.showwarning("Edit", "Invalid numeric value.")
            return
        self.peaks.sort(key=lambda p: p.center)
        self.refresh_tree(keep_selection=True)
        self.refresh_plot()

    def delete_selected(self):
        sel = self._selected_index()
        if sel is None:
            return
        del self.peaks[sel]
        self.refresh_tree()
        self.refresh_plot()

    def clear_peaks(self):
        self.peaks.clear()
        self.refresh_tree()
        self.refresh_plot()

    def _selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        return int(self.tree.item(sel[0], "values")[0]) - 1

    def refresh_tree(self, keep_selection: bool = False):
        prev = self._selected_index() if keep_selection else None
        desired = max(6, len(self.peaks))
        try:
            self.tree.configure(height=desired)
        except Exception:
            pass
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, p in enumerate(self.peaks, 1):
            self.tree.insert(
                "", "end",
                values=(i,
                        f"{p.center:.6g}",
                        f"{p.height:.6g}",
                        f"{p.fwhm:.6g}",
                        "Yes" if p.lock_width else "No",
                        "Yes" if p.lock_center else "No")
            )
        if keep_selection and prev is not None and 0 <= prev < len(self.peaks):
            kids = self.tree.get_children()
            if prev < len(kids):
                self.tree.selection_set(kids[prev])

    # ----- Fit & Export -----
    def _sync_selected_edits(self):
        sel = self._selected_index()
        if sel is None:
            return
        try:
            pk = self.peaks[sel]
            pk.center = float(self.center_var.get())
            pk.height = float(self.height_var.get())
            pk.fwhm   = max(float(self.fwhm_var.get()), 1e-6)
            pk.lock_width  = bool(self.lockw_var.get())
            pk.lock_center = bool(self.lockc_var.get())
            self.peaks.sort(key=lambda p: p.center)
        except Exception:
            pass

    def step_once(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            return
        self._sync_selected_edits()

        y_target = self.get_fit_target()
        mask = self.current_fit_mask()
        if mask is None or not np.any(mask):
            messagebox.showwarning("Step", "Fit range is empty. Use 'Full range' or set a valid Min/Max.")
            return
        x_fit = self.x[mask]
        y_fit = y_target[mask]

        base_applied = self.use_baseline.get() and self.baseline is not None
        add_mode = (self.baseline_mode.get() == "add")
        base_fit = self.baseline[mask] if (base_applied and add_mode) else None
        mode = "add" if add_mode else "subtract"

        options = self._solver_options()
        solver = self.solver_choice.get()
        options["solver"] = solver
        # Keep solver_options mirrored in persistent cfg for uncertainty paths
        try:
            self.cfg["solver_options"] = dict(options)
            save_config(self.cfg)
        except Exception:
            pass

        self.set_busy(True, "Stepping…")
        payload = {
            "x": x_fit,
            "y": y_fit,
            "peaks": self.peaks,
            "mode": mode,
            "baseline": base_fit,
            "options": options,
        }
        try:
            if solver == "classic":
                theta, res = classic_step(payload)
            elif solver == "modern_trf":
                theta, res = modern_trf_step(payload)
            elif solver == "modern_vp":
                theta, res = modern_vp_step(payload)
            elif solver == "lmfit_vp":
                theta, res = lmfit_step(payload)
            else:  # pragma: no cover - unknown solver
                raise ValueError(f"unknown solver '{solver}'")
        except Exception as e:  # pragma: no cover - UI feedback only
            self.set_busy(False, "Step failed.")
            messagebox.showerror("Step", f"Step failed:\n{e}")
            self.log(f"Step failed: {e}", level="ERROR")
            return

        if res.accepted:
            j = 0
            for pk in self.peaks:
                c, h, w, eta = theta[j:j+4]; j += 4
                if not pk.lock_center:
                    pk.center = float(c)
                pk.height = float(h)
                if not pk.lock_width:
                    pk.fwhm = float(w)
                pk.eta = float(eta)
            self.refresh_tree(keep_selection=True)
            self.refresh_plot()
            msg = (
                f"Step accepted (Δcost={res.cost0 - res.cost1:.3g}, "
                f"λ={res.lambda_used if res.lambda_used is not None else 'n/a'}, "
                f"backtracks={res.backtracks})"
            )
        else:
            msg = f"Step rejected: {res.reason}"

        self.set_busy(False, msg)
        self.log(msg)

    def fit(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            return
        self._sync_selected_edits()

        x, y, base, mask, mode = self._fit_inputs()
        if mask is None or not np.any(mask):
            messagebox.showwarning("Fit", "Fit range is empty. Use 'Full range' or set a valid Min/Max.")
            return

        # The fit batch runs for each file (game plan 2.1): the same call, the
        # same settings and, with Seed all, the same per-file seed.
        settings = self._settings()
        solver = settings["solver"]
        # Keep solver_options mirrored in persistent cfg for uncertainty paths
        try:
            self.cfg["solver_options"] = {**settings[solver], "solver": solver}
            save_config(self.cfg)
        except Exception:
            pass
        seeds = [
            core_peaks.Peak(p.center, p.height, p.fwhm, p.eta, bool(p.lock_center), bool(p.lock_width))
            for p in self.peaks
        ]
        caps = [(bool(getattr(p, "fwhm_cap_enabled", False)), float(getattr(p, "fwhm_cap_value", 50.0))) for p in self.peaks]
        perf_seed = self._perf_seed_value()
        rng_seed = pipeline.file_seed(self.current_file or "", perf_seed) if settings["perf_seed_all"] else None

        def work():
            return fit_api.run_fit_consistent(
                x, y, seeds, settings, base, mode, mask,
                reheight=False, rng_seed=rng_seed, return_jacobian=True, return_predictors=True,
            )

        def done(res, err):
            self.step_btn.config(state=tk.NORMAL)
            if err or res is None:
                self.set_busy(False, "Fit failed.")
                if err:
                    self.log(traceback.format_exception_only(type(err), err)[0].strip(), level="ERROR")
                    messagebox.showerror("Fit", f"Fitting failed:\n{err}")
                return
            if res.get("skip_reason") is not None:
                self.set_busy(False, "Fit not run.")
                messagebox.showwarning("Fit", f"Fit not run: {res.get('message', '')}")
                return
            self.peaks[:] = [
                Peak(center=pk.center, height=pk.height, fwhm=pk.fwhm, eta=pk.eta,
                     lock_width=pk.lock_width, lock_center=pk.lock_center,
                     fwhm_cap_enabled=cap_on, fwhm_cap_value=cap_value)
                for pk, (cap_on, cap_value) in zip(res["peaks_out"], caps)
            ]
            self._fit = {
                "result": res, "settings": settings, "x": np.array(x), "y": np.array(y),
                "baseline": np.array(base), "mask": np.array(mask), "mode": mode, "peaks": self._peak_state(),
                "rng_seed": rng_seed,
            }
            self.refresh_tree(keep_selection=True)
            self.last_uncertainty = None
            self.ci_band = None
            self.show_ci_band_var.set(False)
            self._last_unc_locks = []
            self._set_ci_toggle_state(False)
            self.refresh_plot()
            rmse = float(res["rmse"])
            self.set_busy(False, f"Fit done. RMSE {rmse:.4g}")
            npts = int(np.count_nonzero(mask))
            self.log(
                f"Fit finished: RMSE={rmse:.4g} over {npts} pts (peaks={len(self.peaks)})"
            )

        self.step_btn.config(state=tk.DISABLED)
        self.set_busy(True, "Fitting…")
        self.run_in_thread(work, done)

    def run_batch(self):
        folder = filedialog.askdirectory(title="Select folder to batch process")
        if not folder:
            return
        out_csv = filedialog.asksaveasfilename(
            title="Save batch summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="batch_summary.csv",
        )
        if not out_csv:
            return
        # Only a trailing .csv is dropped: a typed name keeps its dots.
        base = _dio._drop_csv_suffix(out_csv)
        output_dir = base.parent
        output_base = base.name
        fit_path = output_dir / f"{output_base}_fit.csv"
        unc_path = output_dir / f"{output_base}_uncertainty.csv"
        patterns = [p.strip() for p in self.batch_patterns.get().split(";") if p.strip()]
        if not patterns:
            patterns = ["*.csv", "*.txt", "*.dat"]
        patterns = [str(Path(folder) / p) for p in patterns]
        source = self.batch_source.get()
        peaks_list = []
        if source == "template":
            tname = self.template_var.get()
            tmpl = self.cfg.get("templates", {}).get(tname)
            if not tmpl:
                messagebox.showwarning("Batch", "Select a valid template first.")
                return
            peaks_list = tmpl
        elif source == "current":
            peaks_list = [p.__dict__ for p in self.peaks]

        knobs_raw = {
            "walkers": self._safe_bayes_walkers(),
            "burn": self._safe_bayes_burn(),
            "steps": self._safe_bayes_steps(),
            "thin": self._safe_bayes_thin(),
        }
        try:
            bayes_knobs = validate_unc_knobs(knobs_raw)
        except ValueError as exc:
            self.status_error(f"Bayesian settings invalid: {exc}")
            return
        try:
            self.bayes_walkers_var.set(bayes_knobs["walkers"])
            self.bayes_burn_var.set(bayes_knobs["burn"])
            self.bayes_steps_var.set(bayes_knobs["steps"])
            self.bayes_thin_var.set(bayes_knobs["thin"])
        except Exception:
            pass

        # The settings the GUI fits and exports with, plus the batch's own choices.
        cfg = {
            **self._settings(),
            "peaks": peaks_list,
            "save_traces": bool(self.batch_save_traces.get()),
            "source": source,
            "reheight": bool(self.batch_reheight.get()),
            "auto_max": int(self.batch_auto_max.get()),
            "output_dir": str(output_dir),
            "output_base": output_base,
        }

        self.cfg["batch_patterns"] = self.batch_patterns.get()
        self.cfg["batch_source"] = source
        self.cfg["batch_reheight"] = bool(self.batch_reheight.get())
        self.cfg["batch_auto_max"] = int(self.batch_auto_max.get())
        self.cfg["batch_save_traces"] = bool(self.batch_save_traces.get())
        try:
            val = bool(self.batch_unc_enabled.get())
        except Exception:
            try:
                val = bool(self.compute_uncertainty_batch.get())
            except Exception:
                val = True
        self.cfg["batch_compute_uncertainty"] = val
        self.cfg["compute_uncertainty_batch"] = val
        save_config(self.cfg)

        # Respect the UI toggle if present; otherwise default ON to avoid silent skips.
        compute_unc = True
        try:
            compute_unc = bool(self.batch_unc_enabled.get())
        except Exception:
            try:
                compute_unc = bool(self.compute_uncertainty_batch.get())
            except Exception:
                compute_unc = True
        unc_method = str(self.unc_method.get()).strip()

        self.log("Starting batch…")
        # Helpful visibility when debugging batch behavior:
        self.log(f"Batch: compute_uncertainty={compute_unc} | method={unc_method}")
        abort_evt = self._job_begin()

        def work():
            def prog(i, total, path):
                self.call_on_main(lambda: self.status_var.set(f"Batch {i}/{total}: {Path(path).name}"))

            return batch_runner.run_batch(
                patterns,
                cfg,
                compute_uncertainty=compute_unc,
                unc_method=unc_method,
                progress=prog,
                log=self.log_threadsafe,
                abort_event=abort_evt,
            )

        def done(res, err):
            if self._closing:
                return
            if err or res is None:
                self.set_busy(False, "Batch failed.")
                if err:
                    self.log(f"Batch failed: {err}", level="ERROR")
                    messagebox.showerror("Batch", f"Batch failed:\n{err}")
                return
            if isinstance(res, dict) and res.get("aborted"):
                self.set_busy(False, "Batch aborted.")
                self.log("Batch aborted by user.")
                return
            ok, total = res
            self.set_busy(False, f"Batch done. {ok}/{total} succeeded.")
            self.log(f"Batch done: {ok}/{total} succeeded.")
            messagebox.showinfo("Batch", f"Summary saved:\n{fit_path}")

        self.set_busy(True, "Batch running…")
        self.run_in_thread(work, self._ending(abort_evt, done))

    # --- BEGIN: local helper for canonical method labels ---
    def _unc_method_label(self, info: dict) -> str:
        """Return a stable canonical label for an uncertainty method key."""
        m = str((info or {}).get("method", "")).strip().lower()
        label = "unknown"
        if m.startswith("asymptotic"):
            label = "Asymptotic (JᵀJ)"
        elif m.startswith("bootstrap"):
            label = "Bootstrap"
        elif m.startswith("bayes"):
            label = "Bayesian (MCMC)"
        try:
            if isinstance(label, str) and label.lower().startswith("bayes"):
                label += "\n(ESS↑ and R̂→1 indicate better mixing; MCSE shows Monte Carlo error of reported quantiles.)"
        except Exception:
            pass
        return label
    # --- END: local helper ---

    def _unc_selected_method_key(self) -> str:
        """Return the canonical uncertainty method key."""
        return method_display_to_key(self.unc_method.get())

    # --- Uncertainty: the one context batch uses (game plan 2.1) ---------------
    def _uncertainty_key(self, settings: dict) -> dict:
        """The settings an uncertainty result depends on: all but export-only ones."""

        return {k: v for k, v in settings.items() if k not in ("export_unc_wide", "log_debug_mode")}

    def _cached_uncertainty(self, rec, settings: dict):
        """The uncertainty already computed for the fit ``rec`` with these settings."""

        cache = getattr(self, "_unc_cache", None)
        if cache and rec is not None and cache["fit"] is rec and cache["key"] == self._uncertainty_key(settings):
            return cache["result"]
        return None

    @staticmethod
    def _uncertainty_blocker(rec) -> Optional[str]:
        """Why no uncertainty can be computed now, or None.

        Like batch, the GUI computes the uncertainty of a converged fit, and of the
        peaks on screen only while they are that fit's."""

        if rec is None:
            return (
                "Fit first. No fit has produced these peaks on these data, baseline and "
                "window: they are new, or changed since the last fit."
            )
        if not rec["result"].get("fit_ok"):
            return "the fit did not converge. Fit again, or change the solver settings."
        return None

    def _uncertainty_settings_ok(self, settings: dict) -> bool:
        if settings.get("unc_method") != "bayesian":
            return True
        try:
            validate_unc_knobs(
                {k: settings[f"bayes_{k}"] for k in ("walkers", "burn", "steps", "thin")}
            )
        except ValueError as exc:
            self.status_error(f"Bayesian settings invalid: {exc}")
            return False
        return True

    def _compute_uncertainty(self, rec: dict, settings: dict, abort_event=None):
        """The uncertainty of the fit ``rec``, on a worker thread: no Tk calls here.

        Batch's context, unchanged: the window's points, the fit's baseline, locks,
        bounds and noise weights, and refits with the fit's own settings and seed.
        Returns the method's raw result and the finished one the writers take; a run
        that ``abort_event`` stopped raises pipeline.UncertaintyAborted instead."""

        method = settings["unc_method"]
        raw = pipeline.compute_uncertainty(
            rec["result"], rec["x"], rec["y"], rec["mask"], settings, method,
            fit_config=rec["settings"], file_seed=rec.get("rng_seed"),
            abort_event=abort_event, log=self.log_threadsafe,
        )
        return raw, pipeline.finish_uncertainty(raw, rec["result"], rec["settings"], method)

    @staticmethod
    def _uncertainty_error_text(error) -> str:
        msg = str(error)
        diag = getattr(error, "diagnostics", None) or {}
        if "Insufficient successful bootstrap refits" in msg:
            errs = diag.get("refit_errors")
            if errs:
                msg = f"{msg} | {' | '.join(map(str, errs[:5]))}"
        return msg

    def _show_uncertainty(self, rec: dict, settings: dict, raw, unc_norm: dict) -> None:
        """Keep a computed uncertainty for Export and show it (Tk thread)."""

        self._unc_cache = {"fit": rec, "key": self._uncertainty_key(settings), "result": unc_norm}
        self.last_uncertainty = unc_norm
        self._last_unc_locks = pipeline.locks(rec["result"])
        band = self._band_tuple_from(unc_norm)
        self.ci_band = band
        self._set_ci_toggle_state(True)
        # A band just became available: honour the user's saved per-method preference.
        try:
            mk = self._unc_selected_method_key()
            if (
                band is not None
                and bool(getattr(self, "_band_pref", {}).get(mk, False))
                and not bool(self.show_ci_band_var.get())
            ):
                try:
                    self._suspend_ci_trace = True
                    self.show_ci_band_var.set(True)
                finally:
                    self._suspend_ci_trace = False
                self._toggle_ci_band()
        except Exception:
            pass
        label = str(unc_norm.get("label") or self._unc_method_label({"method": settings.get("unc_method")}))
        self.status_info(f"Computed {label} uncertainty.")
        diag = (raw.get("diagnostics") if isinstance(raw, dict) else getattr(raw, "diagnostics", None)) or {}
        if label.lower().startswith("bayes"):
            self._report_bayes_diagnostics(diag, band)
        n_fail = int(diag.get("n_fail") or 0)
        if n_fail:
            self.status_warn(f"{n_fail} of {int(diag.get('n_boot') or 0)} bootstrap refits failed and were left out.")
        for i, row in enumerate(unc_norm.get("stats") or [], start=1):
            def _fmt(blk):
                blk = blk if isinstance(blk, dict) else {}
                try:
                    est = f"{float(blk.get('est')):.6g}"
                except (TypeError, ValueError):
                    est = "n/a"
                try:
                    sd = f"{float(blk.get('sd')):.3g}"
                except (TypeError, ValueError):
                    sd = "n/a"
                return f"{est} ± {sd}"

            self.status_info(
                f"Peak {i}: center={_fmt(row.get('center'))} | height={_fmt(row.get('height'))} | "
                f"FWHM={_fmt(row.get('fwhm'))} | eta={_fmt(row.get('eta'))}"
            )
        self.refresh_plot()

    def _report_bayes_diagnostics(self, diag: dict, band) -> None:
        if bool(diag.get("diagnostics_enabled")):
            self.log_threadsafe(
                f"Posterior health: ess_min={_finite_or_nan(diag.get('ess_min')):.3g}, "
                f"rhat_max={_finite_or_nan(diag.get('rhat_max')):.3g}, "
                f"MCSE_mean={_finite_or_nan(diag.get('mcse_mean')):.3g}"
            )
        if diag.get("chain_short"):
            ratio = diag.get("chain_tau_ratio")
            self.status_warn(
                f"The MCMC chain is {float(ratio):.1f} autocorrelation times long; emcee advises at least "
                f"{float(diag.get('chain_tau_min') or 50):g}: run more steps."
                if ratio is not None and math.isfinite(float(ratio))
                else "The MCMC autocorrelation time could not be estimated: run more steps."
            )
        if band is None and bool(diag.get("band_gated")):
            skip = diag.get("band_skip_reason")
            if skip == BAND_SKIP_DIAG_DISABLED:
                self.status_warn("Bayesian band not computed: diagnostics were OFF (no force).")
            elif skip == BAND_SKIP_DIAG_UNHEALTHY:
                thr = diag.get("band_gating_thresholds", {})
                self.status_warn(
                    f"Bayesian band gated: ESS_min={_finite_or_nan(thr.get('ess_min')):.3g} (≥{thr.get('ess_min_req')}), "
                    f"R̂_max={_finite_or_nan(thr.get('rhat_max')):.3g} (≤{thr.get('rhat_max_req')}), "
                    f"MCSE_mean={_finite_or_nan(thr.get('mcse_mean')):.3g} (≤{thr.get('mcse_mean_req')})."
                )
            elif skip == BAND_SKIP_INSUFF_DRAWS:
                self.log_threadsafe(
                    "Bayesian band skipped: not enough posterior draws to build a stable band. "
                    "Try increasing walkers/steps or reducing thinning; alternatively lower the band draw cap in Settings."
                )
            elif skip == BAND_SKIP_ABORTED:
                self.status_warn("Bayesian band aborted before completion.")
        elif band is not None and bool(diag.get("band_forced")):
            self.status_info("Bayesian band computed (forced).")

    @log_action
    def run_uncertainty(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            messagebox.showinfo("Uncertainty", "Load data and perform a fit first.")
            return
        self._sync_selected_edits()
        rec = self._current_fit()
        blocker = self._uncertainty_blocker(rec)
        if blocker is not None:
            self.status_warn(f"Uncertainty not computed: {blocker}")
            return
        settings = self._settings()
        if not self._uncertainty_settings_ok(settings):
            return
        self._unc_job_id += 1
        job_id = self._unc_job_id
        abort_evt = self._job_begin()

        def work():
            return self._compute_uncertainty(rec, settings, abort_evt)

        def done(result, error):
            self._progress_end("uncertainty")
            if job_id != self._unc_job_id or self._closing:
                return  # a newer run replaced this one, or the window is closing
            self._unc_running = False
            if isinstance(error, pipeline.UncertaintyAborted):
                # Nothing is shown or kept: the draws made so far are no result.
                self.status_warn("Uncertainty aborted: no result was kept.")
                return
            if error is not None:
                self.status_error(f"Uncertainty failed: {self._uncertainty_error_text(error)}")
                return
            raw, unc_norm = result
            self._show_uncertainty(rec, settings, raw, unc_norm)

        self._unc_running = True
        self._progress_begin("uncertainty")
        self.status_info("Computing uncertainty…")
        self.run_in_thread(work, self._ending(abort_evt, done))


    # Performance apply scheduler (debounce for trace-driven calls to reduce log spam)
    def request_apply_performance(
        self, 
        source: str = "unknown", 
        immediate: bool = False, 
        save_cfg: bool = True,
        force_log: bool = False
    ) -> None:
        """
        Request a performance apply with debounce for trace-driven calls.
        
        Args:
            source: Where this request came from (e.g., "trace:seed_var", "button:Apply")
            immediate: If True, run immediately and cancel any pending call
            save_cfg: Whether to persist the settings. A debounced request saves
                them at once and leaves only the apply and its log line for later.
            force_log: Force logging of backend line even if unchanged
            
        Side-effects:
            Schedules or executes apply_performance with appropriate parameters
        """
        
        # Cancel any pending apply
        if self._perf_apply_after_id is not None:
            try:
                self.root.after_cancel(self._perf_apply_after_id)
            except Exception:
                pass
            self._perf_apply_after_id = None
        
        if immediate:
            # Run now, no debounce
            self._apply_performance_now(source=source, save_cfg=save_cfg, force_log=force_log)
            return
        
        # Save now: a window closed inside the debounce window must not lose the change
        if save_cfg:
            self._store_perf_settings()
            save_config(self.cfg)

        # Debounce with 150ms window - multiple rapid calls collapse to one
        def _do_apply():
            self._perf_apply_after_id = None
            self._perf_pending_source = source
            self._apply_performance_now(source=source, save_cfg=False, force_log=force_log)
        
        self._perf_apply_after_id = self.root.after(150, _do_apply)

    def _compute_perf_signature(self) -> tuple:
        """Compute a signature for current performance state (for log dedupe)."""
        try:
            from infra import performance as perf_mod
            
            backend_name = perf_mod.which_backend()
            unc_req = int(self.perf_unc_workers.get()) if hasattr(self, "perf_unc_workers") else 0
            strategy = str(self.perf_parallel_strategy.get()) if hasattr(self, "perf_parallel_strategy") else "outer"
            
            mkl_threads = os.environ.get("MKL_NUM_THREADS", "-")
            openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "-")
            omp_threads = os.environ.get("OMP_NUM_THREADS", "-")
            
            seed_all = bool(self.perf_seed_all.get()) if hasattr(self, "perf_seed_all") else False
            seed_val = None
            try:
                if seed_all and hasattr(self, "seed_var"):
                    seed_txt = str(self.seed_var.get()).strip()
                    seed_val = int(seed_txt) if seed_txt else 0
            except Exception:
                pass
            
            return (backend_name, unc_req, strategy, mkl_threads, openblas_threads, omp_threads, seed_all, seed_val)
        except Exception:
            # Fallback signature on error
            return ("error", 0, "unknown", "-", "-", "-", False, None)

    
    def apply_performance(
        self, 
        source: str = "unknown", 
        immediate: bool = True, 
        save_cfg: bool = True,
        force_log: bool = False
    ) -> None:
        """Public wrapper for backward compatibility."""
        
        # For explicit calls (buttons), use immediate=True; traces should call request_apply_performance directly
        self.request_apply_performance(
            source=source if not immediate else "explicit", 
            immediate=immediate, 
            save_cfg=save_cfg, 
            force_log=force_log
        )

    def _store_perf_settings(self) -> dict:
        """Copy the performance settings from the widgets into self.cfg and return them."""
        try:
            seed_txt = str(self.seed_var.get()).strip()
        except Exception:
            seed_txt = ""
        seed = None
        if seed_txt:
            try:
                seed = int(seed_txt)
            except Exception:
                seed = None
        # Parse workers safely even if the Entry is blank while typing
        try:
            unc_txt = str(self.perf_unc_workers.get()).strip()
        except Exception:
            unc_txt = ""
        try:
            unc_workers = int(float(unc_txt)) if unc_txt else 0
        except Exception:
            unc_workers = 0
        settings = {
            "perf_numba": bool(self.perf_numba.get()),
            "perf_cache_baseline": bool(self.perf_cache_baseline.get()),
            "perf_seed_all": bool(self.perf_seed_all.get()),
            "perf_seed": seed if seed is not None else "",
            "perf_unc_workers": unc_workers,
            "perf_parallel_strategy": str(self.perf_parallel_strategy.get()),
            "perf_blas_threads": int(self.perf_blas_threads.get()),
            "log_debug_mode": bool(self.debug_mode_var.get()),
        }
        self.cfg.update(settings)
        return settings

    def _apply_performance_now(
        self, 
        source: str = "unknown", 
        save_cfg: bool = True, 
        force_log: bool = False
    ) -> None:
        """Internal core apply with logging dedupe logic."""
        
        # Call the actual apply logic
        settings = self._store_perf_settings()
        seed = settings["perf_seed"] if settings["perf_seed"] != "" else None
        unc_workers = settings["perf_unc_workers"]
        performance.set_numba(settings["perf_numba"])
        performance.set_cache_baseline(settings["perf_cache_baseline"])
        # apply_global_seed stores the seed and, if enabled, seeds RNGs globally.
        # No need to call set_seed separately.
        performance.apply_global_seed(seed, settings["perf_seed_all"])
        performance.set_unc_workers(unc_workers)

        # Set env for child processes/futures
        import os

        strategy = settings["perf_parallel_strategy"]
        blas_threads = settings["perf_blas_threads"]
        if strategy == "outer":
            self._apply_thread_env("1", "1", "1")
        else:
            if blas_threads > 0:
                desired = str(blas_threads)
                self._apply_thread_env(desired, desired, desired)
            else:
                for k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
                    os.environ.pop(k, None)

        # Log effective libs and clamp via threadpoolctl when available
        pre_info = threadpool_info()
        self.dlog(f"[DEBUG] threadpool_info_pre={pre_info}")
        _set_thread_limits(strategy, blas_threads)
        post_info = threadpool_info()
        self.dlog(f"[DEBUG] threadpool_info_post={post_info}")
        libs = post_info
        self.dlog(f"threadpool_info={libs}")
        if save_cfg:
            save_config(self.cfg)
        try:
            unc_req = int(self.perf_unc_workers.get())
        except Exception:
            unc_req = unc_workers
        # what the uncertainty runs with: "auto" (0) is one worker, i.e. sequential
        unc_eff = pipeline.uncertainty_workers({"perf_unc_workers": unc_req})[0]
        self.dlog("[DEBUG] apply_performance()")
        # Log only if the applied state changed or the caller forces it, to reduce
        # spam. The signature is taken after the apply, so it holds the new backend.
        new_signature = self._compute_perf_signature()
        if force_log or new_signature != self._perf_last_signature:
            self.log(
                "Backend: %s | unc_workers_req=%s | unc_workers_eff=%s | "
                "strategy=%s | MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s | seed_all=%s seed=%s"
                % (
                    performance.which_backend(),
                    unc_req,
                    unc_eff,
                    strategy,
                    os.environ.get("MKL_NUM_THREADS", "-"),
                    os.environ.get("OPENBLAS_NUM_THREADS", "-"),
                    os.environ.get("OMP_NUM_THREADS", "-"),
                    bool(self.perf_seed_all.get()),
                    seed if seed is not None else "-",
                )
            )
        else:
            self.dlog("[DEBUG] apply_performance skipped - no signature change")
        self._perf_last_signature = new_signature
        self.status_var.set("Performance options applied.")

    # minimize env toggling across older Python/BLAS/OpenMP combos
    def _set_env_if_changed(self, key: str, val: str):
        try:
            cur = os.environ.get(key)
            if cur != val:
                os.environ[key] = val
        except Exception:
            pass

    def _apply_thread_env(self, mkl: str, openblas: str, omp: str):
        self._set_env_if_changed("MKL_NUM_THREADS", mkl)
        self._set_env_if_changed("OPENBLAS_NUM_THREADS", openblas)
        self._set_env_if_changed("OMP_NUM_THREADS", omp)

    def on_export(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            messagebox.showinfo("Export", "Load data and perform a fit first.")
            return
        if getattr(self, "_export_running", False):
            self.status_warn("Export is already running.")
            return
        out_csv = filedialog.asksaveasfilename(
            title="Save export base name",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out_csv:
            return
        self._sync_selected_edits()
        # Only a trailing .csv is dropped: a typed MoS2_0.5mW keeps its dots.
        base = _dio._drop_csv_suffix(out_csv)
        # The fit the table describes, with the settings it ran with; peaks no fit
        # produced (placed by hand, or edited since the fit) say fit_ok False.
        rec = self._current_fit()
        fit = self._export_fit()
        settings = self._settings()
        write_settings = {**(rec["settings"] if rec is not None else settings), "export_unc_wide": settings["export_unc_wide"]}
        blocker = self._uncertainty_blocker(rec)
        if blocker is None and not self._uncertainty_settings_ok(settings):
            blocker = "the uncertainty settings are invalid."
        cached = self._cached_uncertainty(rec, settings)
        source = str(self.current_file or "")
        keep = {os.path.normcase(os.path.abspath(source))} if source else set()
        x, y_raw = self.x, self.y_raw
        abort_evt = self._job_begin()

        def work():
            # A worker thread: the uncertainty, then the files, and no Tk calls.
            unc, computed, unc_error = cached, None, None
            if blocker is None and cached is None:
                try:
                    computed = self._compute_uncertainty(rec, settings, abort_evt)
                    unc = computed[1]
                except Exception as exc:
                    unc_error = exc
            # The writer batch uses: the fit table, the trace and the uncertainty
            # files, after removing uncertainty files an earlier export left here.
            written = pipeline.write_spectrum(
                base, source, x=x, y_raw=y_raw, fit=fit, settings=write_settings,
                unc_norm=unc, trace=True, keep=keep, log=self.log_threadsafe,
            )
            return written, unc, computed, unc_error

        def done(result, error):
            self._export_running = False
            self.set_busy(False, "Export done." if error is None else "Export failed.")
            if error is not None:
                self.log(f"Export failed: {error}", level="ERROR")
                if not self._closing:
                    messagebox.showerror("Export", f"Export failed:\n{error}")
                return
            written, unc, computed, unc_error = result
            if computed is not None and self._current_fit() is rec and not self._closing:
                self._show_uncertainty(rec, settings, computed[0], computed[1])
            if blocker is not None:
                self.status_warn(f"Uncertainty not computed: {blocker}")
            elif isinstance(unc_error, pipeline.UncertaintyAborted):
                self.status_warn("Uncertainty not exported: the run was aborted. Wrote the fit and the trace only.")
            elif unc_error is not None:
                self.status_warn(f"Uncertainty export failed: {self._uncertainty_error_text(unc_error)}")
            else:
                self.status_info(
                    "Exporting uncertainty: reusing the computed result." if cached is not None
                    else "Exporting uncertainty: computed for this export."
                )
                self.log(f"Exported uncertainty ({unc['label']}).")
            saved_lines = [str(p) for p in written]
            for p in saved_lines:
                self.log(f"Export: wrote {p}")
            if not self._closing:
                messagebox.showinfo("Export", "Saved:\n" + "\n".join(saved_lines))

        self._export_running = True
        self.set_busy(True, "Exporting…")
        self.run_in_thread(work, self._ending(abort_evt, done))


    # ----- Plot -----
    def toggle_components(self):
        self.components_visible = not self.components_visible
        self.cfg["ui_show_components"] = self.components_visible
        save_config(self.cfg)
        self.refresh_plot()

    def refresh_plot(self):
        if not hasattr(self, "ax") or self.ax is None:
            return
        LW_RAW, LW_BASE, LW_CORR, LW_COMP, LW_FIT = 1.0, 1.0, 0.9, 0.8, 1.2
        self.ax.clear()
        self.ax.set_xlabel(format_axis_label_inline(self.x_label_var.get(), self.x_label_auto_math.get()))
        self.ax.set_ylabel("Intensity")
        if self.x is None:
            self.ax.set_title("Open a data file to begin")
            self.canvas.draw_idle()
            return

        base_applied = self.use_baseline.get() and self.baseline is not None
        add_mode = (self.baseline_mode.get() == "add")
        base = self.baseline if base_applied else np.zeros_like(self.x)

        self.ax.plot(self.x, self.y_raw, lw=LW_RAW, label="Raw")
        if base_applied:
            self.ax.plot(self.x, base, lw=LW_BASE, label="Baseline")
            if not add_mode:
                self.ax.plot(self.x, self.y_raw - base, lw=LW_CORR, label="Corrected")

        if self.peaks:
            total_peaks = np.zeros_like(self.x)
            if self.components_visible:
                sig = int(self.legend_center_sigfigs.get())
                for i, p in enumerate(self.peaks, 1):
                    comp = pseudo_voigt(self.x, p.height, p.center, p.fwhm, p.eta)
                    total_peaks += comp
                    comp_plot = (base + comp) if (base_applied and add_mode) else comp
                    label = f"Peak {i} @ {p.center:.{sig}g}"
                    self.ax.plot(self.x, comp_plot, lw=LW_COMP, alpha=0.6, label=label)
            else:
                for p in self.peaks:
                    total_peaks += pseudo_voigt(self.x, p.height, p.center, p.fwhm, p.eta)
            total = (base + total_peaks) if (base_applied and add_mode) else total_peaks
            self.ax.plot(self.x, total, "--", lw=LW_FIT, label="Total fit")
            for p in self.peaks:
                self.ax.axvline(p.center, color="k", linestyle=":", alpha=0.35, lw=0.7)

        if self.fit_xmin is not None and self.fit_xmax is not None:
            lo, hi = sorted((self.fit_xmin, self.fit_xmax))
            self.ax.axvspan(lo, hi, color="0.8", alpha=0.25, lw=0)

        if self.show_ci_band_var.get() and self.ci_band is not None:
            try:
                xb, lob, hib = self.ci_band
                self.ax.fill_between(
                    np.asarray(xb), np.asarray(lob), np.asarray(hib),
                    alpha=0.18, label="Uncertainty band"
                )
            except Exception as e:
                if hasattr(self, "status_warn"):
                    self.status_warn(f"Uncertainty band unavailable: {type(e).__name__}")

        # Legend toggle
        leg = self.ax.get_legend()
        if self.show_legend_var.get():
            try:
                self.ax.legend(loc="best", prop=FontProperties(family="Arial"))
            except Exception:
                pass
        else:
            if leg is not None:
                try:
                    leg.remove()
                except Exception:
                    pass
        self.canvas.draw_idle()

    # ----- Help -----
    def show_help(self):
        from . import helptext

        opts = {
            "modern_losses": MODERN_LOSSES,
            "modern_weights": MODERN_WEIGHTS,
            "lmfit_algos": LMFIT_ALGOS,
        }
        message = helptext.build_help(opts)
        win = tk.Toplevel(self.root)
        win.title("Help")
        txt = tk.Text(win, wrap="word", width=100)
        txt.insert("1.0", message)
        txt.config(state="disabled")
        scroll = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)




    def _apply_performance_immediate(self) -> None:
        """Helper for Apply button - always runs immediately with logging."""
        self._apply_performance_now(source="button:Apply", save_cfg=True, force_log=True)

def main():
    import sys
    
    root = tk.Tk()
    
    # Apply modern theme with graceful fallback to default Tkinter
    theme_applied, theme_name = apply_theme(root)
    log_theme_status(theme_applied, theme_name)
    
    # --- Font policy: configure before creating PeakFitApp ---
    from ui.fonts import (pick_first_available_family, 
                          configure_named_fonts, 
                          configure_ttk_styles, 
                          DEFAULT_UI_SIZE,
                          SANS_CANDIDATES,
                          MONO_CANDIDATES)
    
    _style = ttk.Style(root)
    _cfg = load_config()
    
    # Diagnostics first, so they report the fonts before the policy changes them.
    # Asked for by the --font-diagnostics flag or the font_diagnostics config key.
    if "--font-diagnostics" in sys.argv or bool(_cfg.get("font_diagnostics", False)):
        from ui.font_diagnostics import run_diagnostics as diag_run, show_visual_sample
        diag_run(root, log_fn=print)
        show_visual_sample(root)
    
    # Determine font families (respect config or use auto selection)
    _sans_auto = _cfg.get("ui_font_family", "auto")
    _mono_auto = _cfg.get("ui_mono_font_family", "auto")
    
    if _sans_auto == "auto":
        _sans_selected = pick_first_available_family(root, SANS_CANDIDATES)
    else:
        _sans_selected = _sans_auto
    
    if _mono_auto == "auto":
        _mono_selected = pick_first_available_family(root, MONO_CANDIDATES)
    else:
        _mono_selected = _mono_auto
        
    _size = int(_cfg.get("ui_font_size", DEFAULT_UI_SIZE))
    
    configure_named_fonts(root, sans_family=_sans_selected, 
                         mono_family=_mono_selected, ui_size=_size)
    configure_ttk_styles(_style, sans_family=_sans_selected, 
                        mono_family=_mono_selected, ui_size=_size)
    # --- END font policy ---
    
    app = PeakFitApp(root, cfg=_cfg)  # one read per start: a file set aside stays aside
    root.mainloop()


if __name__ == "__main__":
    main()
