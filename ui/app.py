#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive peak fit GUI for spectra (Gaussian–Lorentzian / pseudo-Voigt)
Designed by Farhan Zahin • Built with ChatGPT

Build: v3.2-beta

What’s new since v3.0 / v3.1
• Reliability & parity: Classic is restored as a simple SciPy curve_fit backend (lock-aware, minimal bounds). Modern TRF/VP stabilized (robust losses/weights; VP heights via NNLS).
• “Step ▶” parity: Step uses the same residual/bounds as Fit and reports damping λ, backtracks, step_norm, and accept/reject reasons. Non-finite guards prevent freezes.
• Persisted settings: Shape factor η, “Add peaks on click”, solver choice, uncertainty method, performance toggles, legend visibility, and defaults (e.g., labels) are saved to ~/.gl_peakfit_config.json.
• UI/UX: Segmented action bar (File | Fit | Plot | Help) is always visible; F1 opens Help. Right panel scrolling is independent from the Help window; splitter enforces sensible min widths; status bar with progress + collapsible green-on-black log; batch streams progress.
• Defaults: “Show uncertainty band” = ON. “Baseline uses fit range” = ON. Performance toggles persist (Numba/CuPy/cache/seed/workers).
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
The configuration file (~/.gl_peakfit_config.json) stores: ALS defaults (λ, p, iterations, threshold), solver choice & options (loss/weights/f_scale/maxfev/restarts/jitter/step_lambda), uncertainty method, click-to-add toggle, global η, performance toggles (numba/gpu/cache/seed_all/max_workers), batch defaults, x-label, legend visibility, and templates (including auto-apply). Settings update on change and are loaded at startup.

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
import math
import re
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import tkinter.font as tkfont
import threading
import traceback

from scipy.signal import find_peaks

from core import signals, data_io
try:
    from core.uncertainty import UncertaintyResult, NotAvailable
except Exception:  # pragma: no cover - NotAvailable may be absent
    from core.uncertainty import UncertaintyResult  # type: ignore

    class NotAvailable:  # pragma: no cover - fallback placeholder
        pass
from core.residuals import build_residual, jacobian_fd
from core.fit_api import (
    classic_step,
    modern_trf_step,
    modern_vp_step,
    lmfit_step,
)
from fit import orchestrator, classic, modern_vp, modern
try:  # optional
    from fit import lmfit_backend
except Exception:  # pragma: no cover - optional dependency may be missing
    lmfit_backend = None

BACKENDS = {
    "classic": classic,
    "modern_vp": modern_vp,
    "modern_trf": modern,
}
if lmfit_backend is not None:
    BACKENDS["lmfit_vp"] = lmfit_backend
from infra import performance
from batch import runner as batch_runner
from uncertainty import asymptotic, bayes, bootstrap

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


# ---------- Config persistence ----------
CONFIG_PATH = Path.home() / ".gl_peakfit_config.json"
DEFAULTS = {
    "als_lam": 1e5,
    "als_asym": 0.001,
    "als_niter": 10,
    "als_thresh": 0.0,
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
    "ui_add_peaks_on_click": True,
    "unc_method": "asymptotic",
    "x_label_auto_math": True,
    "ui_show_legend": True,
    "legend_center_sigfigs": 6,
    # Uncertainty and performance defaults
    "show_uncertainty_band": True,
    "baseline_uses_fit_range": True,
    "perf_numba": False,
    "perf_gpu": False,
    "perf_cache_baseline": True,
    "perf_seed_all": False,
    "perf_max_workers": 0,
}

LOG_MAX_LINES = 5000


def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            cfg = {**DEFAULTS, **data}
            # Migration: move legacy saved_peaks into templates["default"] if templates is empty
            if cfg.get("saved_peaks") and not cfg.get("templates"):
                cfg["templates"] = {"default": cfg["saved_peaks"]}
            # Migration: legacy solver names
            sc = cfg.get("solver_choice")
            if sc == "modern":
                cfg["solver_choice"] = "modern_vp"
            elif sc == "lmfit":
                cfg["solver_choice"] = "lmfit_vp"
            cfg.pop("ui_theme", None)
            return cfg
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        messagebox.showwarning("Config", f"Could not save config: {e}")


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

    from core import data_io

    return data_io.load_xy(path)


def _unc_method_label(res: Any) -> str:
    for k in ("method_label", "label", "method", "type"):
        v = getattr(res, k, None) if not isinstance(res, dict) else res.get(k)
        if isinstance(v, str) and v.strip():
            m = v.strip()
            break
    else:
        return "Unknown"
    m_low = m.lower()
    return {
        "asymptotic": "Asymptotic (JᵀJ)",
        "bootstrap": "Bootstrap (residual)",
        "bayesian": "Bayesian (MCMC)",
    }.get(m_low, m)


def _coerce_param_stats(res: Any) -> Dict[str, Dict[str, Any]]:
    """Return normalized parameter stats mapping."""

    def _norm_map(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        nested = True
        for v in d.values():
            if not isinstance(v, dict):
                nested = False
                break
        if nested:
            for p, inner in d.items():
                if not isinstance(inner, dict):
                    continue
                est = inner.get("est") or inner.get("mean") or inner.get("median")
                sd = inner.get("sd") or inner.get("stderr") or inner.get("sigma")
                p25 = (
                    inner.get("p2_5")
                    or inner.get("p025")
                    or inner.get("q2_5")
                    or inner.get("q025")
                )
                p975 = (
                    inner.get("p97_5")
                    or inner.get("p975")
                    or inner.get("q97_5")
                    or inner.get("q975")
                )
                out[p] = {"est": est, "sd": sd, "p2_5": p25, "p97_5": p975}
            return out

        buckets: Dict[str, Dict[str, Any]] = {}
        for k, v in d.items():
            if not isinstance(k, str) or "_" not in k:
                continue
            base, suffix = k.rsplit("_", 1)
            suffix = suffix.lower()
            if suffix not in (
                "est",
                "mean",
                "median",
                "sd",
                "stderr",
                "sigma",
                "p2",
                "p2_5",
                "p025",
                "p97",
                "p97_5",
                "p975",
                "q2_5",
                "q025",
                "q97_5",
                "q975",
            ):
                continue
            b = buckets.setdefault(base, {})
            b[suffix] = v

        for base, inner in buckets.items():
            est = inner.get("est") or inner.get("mean") or inner.get("median")
            sd = inner.get("sd") or inner.get("stderr") or inner.get("sigma")
            p25 = inner.get("p2_5") or inner.get("p025") or inner.get("p2") or inner.get("q2_5") or inner.get("q025")
            p975 = (
                inner.get("p97_5")
                or inner.get("p975")
                or inner.get("p97")
                or inner.get("q97_5")
                or inner.get("q975")
            )
            out[base] = {"est": est, "sd": sd, "p2_5": p25, "p97_5": p975}
        return out

    if isinstance(res, dict):
        for key in ("param_stats", "params", "parameters", "stats"):
            if key in res and isinstance(res[key], dict):
                return _norm_map(res[key])
        return _norm_map(res)
    else:
        for key in ("param_stats", "params", "parameters", "stats"):
            if hasattr(res, key):
                val = getattr(res, key)
                if isinstance(val, dict):
                    return _norm_map(val)
        if hasattr(res, "to_dict"):
            maybe = res.to_dict()
            if isinstance(maybe, dict):
                return _norm_map(maybe)
    return {}


def _iter_peakwise(stats_map: Dict[str, Dict[str, Any]], n_peaks: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    def _get(param: str):
        return stats_map.get(param) or stats_map.get(param.rstrip("s"))

    rows = []
    for i in range(n_peaks):
        row: Dict[str, Any] = {}
        for pname in ("center", "fwhm", "height"):
            rec = _get(pname)
            if not rec:
                continue

            def pick(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return x[i] if i < len(x) else None
                return x

            row[pname] = {
                "est": pick(rec.get("est")),
                "sd": pick(rec.get("sd")),
                "p2_5": pick(rec.get("p2_5")),
                "p97_5": pick(rec.get("p97_5")),
            }
        rows.append((i + 1, row))
    return rows


def _format_unc_row(i: int, row: Dict[str, Any]) -> str:
    def fmt(name: str) -> str:
        v = row.get(name) or {}
        est = v.get("est")
        sd = v.get("sd")
        if est is None or sd is None:
            return f"{name} = n/a"
        s = f"{name} = {est:.6g} ± {sd:.2g}"
        p25 = v.get("p2_5")
        p975 = v.get("p97_5")
        if p25 is not None and p975 is not None:
            s += f" (2.5%={p25:.6g}, 97.5%={p975:.6g})"
        return s

    parts = [fmt("center"), fmt("fwhm"), fmt("height")]
    return f"Peak {i}: " + " | ".join(parts)


# ---------- Main GUI ----------
class PeakFitApp:
    def __init__(self, root, cfg=None):
        self.root = root
        self.cfg = cfg if cfg is not None else load_config()
        self.cfg.setdefault("baseline_uses_fit_range", True)
        self.cfg.setdefault("ui_show_uncertainty_band", True)
        self.cfg.setdefault("perf_numba", False)
        self.cfg.setdefault("perf_gpu", False)
        self.cfg.setdefault("perf_cache_baseline", True)
        self.cfg.setdefault("perf_seed_all", False)
        self.cfg.setdefault("perf_max_workers", 0)
        self.cfg.setdefault("unc_workers", 0)
        save_config(self.cfg)
        self.root.title("Interactive Peak Fit (pseudo-Voigt)")

        performance.set_logger(self.log_threadsafe)

        self.default_font = tkfont.nametofont("TkDefaultFont")

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

        # Config
        self.als_lam = tk.DoubleVar(value=self.cfg["als_lam"])
        self.als_asym = tk.DoubleVar(value=self.cfg["als_asym"])
        self.als_niter = tk.IntVar(value=self.cfg["als_niter"])
        self.als_thresh = tk.DoubleVar(value=self.cfg["als_thresh"])
        self.global_eta = tk.DoubleVar(value=self.cfg.get("ui_eta", 0.5))
        self.global_eta.trace_add("write", lambda *_: self._on_eta_change())
        self.auto_apply_template = tk.BooleanVar(value=bool(self.cfg.get("auto_apply_template", False)))
        self.auto_apply_template_name = tk.StringVar(value=self.cfg.get("auto_apply_template_name", ""))

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
        self.template_var = tk.StringVar(value=self.auto_apply_template_name.get())

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
        self.bootstrap_solver_choice = tk.StringVar(value=self.solver_choice.get())
        self.bootstrap_solver_label = tk.StringVar(value=SOLVER_LABELS[self.solver_choice.get()])
        self.solver_title = tk.StringVar(value=SOLVER_LABELS[self.solver_choice.get()])
        self.classic_maxfev = tk.IntVar(value=20000)
        self.classic_centers_window = tk.BooleanVar(value=True)
        self.classic_margin = tk.DoubleVar(value=0.0)
        self.classic_fwhm_min = tk.DoubleVar(value=2.0)
        self.classic_fwhm_max = tk.DoubleVar(value=0.5)
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
        self.lmfit_share_fwhm = tk.BooleanVar(value=False)
        self.lmfit_share_eta = tk.BooleanVar(value=False)
        self.snr_text = tk.StringVar(value="S/N: --")

        self.show_ci_band = bool(self.cfg.get("ui_show_uncertainty_band", True))
        self.ci_band = None
        # Uncertainty state
        self.last_unc_result = None
        self.last_unc_method = None

        self.current_file: Optional[Path] = None
        self.show_ci_band_var = tk.BooleanVar(value=self.show_ci_band)
        self.show_ci_band_var.trace_add("write", self._toggle_ci_band)

        # Uncertainty and performance controls
        unc_cfg = self.cfg.get("unc_method", "asymptotic")
        if unc_cfg == "bootstrap":
            unc_label = f"Bootstrap (base solver = {SOLVER_LABELS[self.solver_choice.get()]})"
        elif unc_cfg == "bayesian":
            unc_label = "Bayesian"
        else:
            unc_label = "Asymptotic"
        self.unc_method = tk.StringVar(value=unc_label)
        self.unc_workers_var = tk.IntVar(value=int(self.cfg.get("unc_workers", 0)))
        self.unc_workers_var.trace_add("write", lambda *_: self._on_unc_workers_change())
        self.perf_numba = tk.BooleanVar(value=bool(self.cfg.get("perf_numba", False)))
        self.perf_gpu = tk.BooleanVar(value=bool(self.cfg.get("perf_gpu", False)))
        self.perf_cache_baseline = tk.BooleanVar(value=bool(self.cfg.get("perf_cache_baseline", True)))
        self.perf_seed_all = tk.BooleanVar(value=bool(self.cfg.get("perf_seed_all", False)))
        self.perf_max_workers = tk.IntVar(value=int(self.cfg.get("perf_max_workers", 0)))
        self.perf_numba.trace_add("write", lambda *_: self.apply_performance())
        self.perf_gpu.trace_add("write", lambda *_: self.apply_performance())
        self.perf_cache_baseline.trace_add("write", lambda *_: self.apply_performance())
        self.perf_seed_all.trace_add("write", lambda *_: self.apply_performance())
        self.perf_max_workers.trace_add("write", lambda *_: self.apply_performance())
        self.seed_var = tk.StringVar(value="")
        self.gpu_chunk_var = tk.IntVar(value=262144)

        # UI
        self._build_ui()
        self._new_figure()
        self._update_template_info()
        self.apply_performance()

        # Uncertainty job tracking / last log
        self._unc_job_id = 0
        self._unc_running = False
        self._last_unc_log = None

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
            self.step_btn = ttk.Button(fit_seg, image=self.step_icon, command=self.step_once)
        except Exception:
            self.step_btn = ttk.Button(fit_seg, text="Step", command=self.step_once)
        self.step_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(fit_seg, text="Fit", command=self.fit).pack(side=tk.LEFT, padx=2)

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
        baseline_box = ttk.Labelframe(right, text="Baseline (ALS)"); baseline_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(baseline_box, text="Apply baseline", variable=self.use_baseline, command=self.refresh_plot).pack(anchor="w")

        mode_row = ttk.Frame(baseline_box); mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Add to fit", variable=self.baseline_mode, value="add", command=self.refresh_plot).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Subtract",  variable=self.baseline_mode, value="subtract", command=self.refresh_plot).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(baseline_box, text="Baseline uses fit range", variable=self.baseline_use_range).pack(anchor="w", pady=(2,0))

        row = ttk.Frame(baseline_box); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="λ (smooth):").pack(side=tk.LEFT)
        ttk.Entry(row, width=10, textvariable=self.als_lam).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="p (asym):").pack(side=tk.LEFT)
        ttk.Entry(row, width=10, textvariable=self.als_asym).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(baseline_box); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Iterations:").pack(side=tk.LEFT)
        ttk.Entry(row2, width=5, textvariable=self.als_niter).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="Threshold:").pack(side=tk.LEFT)
        ttk.Entry(row2, width=7, textvariable=self.als_thresh).pack(side=tk.LEFT, padx=4)

        ttk.Button(baseline_box, text="Recompute baseline", command=self.compute_baseline).pack(side=tk.LEFT, pady=2)
        ttk.Button(baseline_box, text="Save as default", command=self.save_baseline_default).pack(side=tk.LEFT, padx=4)
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

        e_center = ttk.Entry(edit, textvariable=self.center_var, width=10)
        e_height = ttk.Entry(edit, textvariable=self.height_var, width=10)
        e_fwhm   = ttk.Entry(edit, textvariable=self.fwhm_var,   width=10)
        ttk.Label(edit, text="Center").grid(row=0, column=0, sticky="e")
        e_center.grid(row=0, column=1, padx=2)
        ttk.Label(edit, text="Height").grid(row=1, column=0, sticky="e")
        e_height.grid(row=1, column=1, padx=2)
        ttk.Label(edit, text="FWHM").grid(row=2, column=0, sticky="e")
        e_fwhm.grid(row=2, column=1, padx=2)

        ttk.Checkbutton(edit, text="Lock width",  variable=self.lockw_var, command=self.on_lock_toggle).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Checkbutton(edit, text="Lock center", variable=self.lockc_var, command=self.on_lock_toggle).grid(row=3, column=1, sticky="w", pady=2)

        btns = ttk.Frame(peaks_box); btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="➕ Add Peak", command=self.add_peak_from_fields).pack(side=tk.LEFT)
        ttk.Button(btns, text="Apply edits", command=self.apply_edits).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete selected", command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear all", command=self.clear_peaks).pack(side=tk.LEFT, padx=4)

        for w in (e_center, e_height, e_fwhm):
            w.bind("<Return>", lambda _e: self.apply_edits())

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
                text=SOLVER_LABELS[key],
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
        self.unc_method_combo = ttk.Combobox(
            unc_box,
            textvariable=self.unc_method,
            state="readonly",
            values=["Asymptotic", "Bootstrap", "Bayesian"],
            width=14,
        )
        self.unc_method_combo.pack(side=tk.LEFT, padx=4)
        self.unc_method_combo.bind("<<ComboboxSelected>>", self._on_unc_method_change)

        solver_labels = [SOLVER_LABELS[k] for k in ["classic", "modern_vp", "modern_trf"]]
        if self.has_lmfit:
            solver_labels.append(SOLVER_LABELS["lmfit_vp"])
        self.bootstrap_solver_combo = ttk.Combobox(
            unc_box,
            textvariable=self.bootstrap_solver_label,
            state="readonly",
            values=solver_labels,
            width=24,
        )
        self.bootstrap_solver_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_bootstrap_solver_change(),
        )
        self.unc_workers_frame = ttk.Frame(unc_box)
        self.unc_workers_label = ttk.Label(self.unc_workers_frame, text="Bootstrap workers:")
        self.unc_workers_label.pack(side=tk.LEFT)
        self.unc_workers_spin = ttk.Spinbox(
            self.unc_workers_frame,
            from_=0,
            to=64,
            textvariable=self.unc_workers_var,
            width=5,
        )
        self.unc_workers_spin.pack(side=tk.LEFT, padx=2)
        ttk.Button(unc_box, text="Run", command=self.run_uncertainty).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            unc_box,
            text="Show uncertainty band",
            variable=self.show_ci_band_var,
        ).pack(anchor="w", padx=4)
        self._update_unc_widgets()

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
        ttk.Checkbutton(perf_box, text="GPU", variable=self.perf_gpu).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Cache baseline", variable=self.perf_cache_baseline).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Seed all", variable=self.perf_seed_all).pack(anchor="w")
        rowp = ttk.Frame(perf_box); rowp.pack(fill=tk.X, pady=2)
        ttk.Label(rowp, text="Seed:").pack(side=tk.LEFT)
        ttk.Entry(rowp, width=8, textvariable=self.seed_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(rowp, text="Max workers:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Spinbox(rowp, from_=0, to=64, textvariable=self.perf_max_workers, width=5).pack(side=tk.LEFT)
        ttk.Label(rowp, text="GPU chunk:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Entry(rowp, width=7, textvariable=self.gpu_chunk_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(rowp, text="Apply", command=self.apply_performance).pack(side=tk.LEFT, padx=4)

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
        self.pbar = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pbar.pack(side=tk.RIGHT, padx=6)
        self.progress = self.pbar
        self._log_console = None
        self._log_visible = False
        self._log_frame = None
        self._log_buffer: list[str] = []
        self._auto_show_log_on_warn = True
        self._auto_show_log_on_error = True

        # Initial peak list height
        self.refresh_tree()

    def _show_solver_opts(self):
        for f in self.solver_frames.values():
            f.pack_forget()
        frame = self.solver_frames.get(self.solver_choice.get())
        if frame:
            frame.pack(side=tk.LEFT, padx=4)

    def _solver_options(self, choice: str | None = None) -> dict:
        solver = choice or self.solver_choice.get()
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
            }
        if solver == "lmfit_vp":
            return {
                "algo": self.lmfit_algo.get(),
                "maxfev": int(self.lmfit_maxfev.get()),
                "share_fwhm": bool(self.lmfit_share_fwhm.get()),
                "share_eta": bool(self.lmfit_share_eta.get()),
            }
        return {
            "maxfev": int(self.classic_maxfev.get()),
            "bound_centers_to_window": bool(self.classic_centers_window.get()),
            "margin_frac": float(self.classic_margin.get()),
            "fwhm_min_factor": float(self.classic_fwhm_min.get()),
            "fwhm_max_factor": float(self.classic_fwhm_max.get()),
            "height_factor": float(self.classic_height_factor.get()),
        }

    def _on_solver_change(self):
        choice = self.solver_choice.get()
        self.solver_title.set(SOLVER_LABELS[choice])
        self.cfg["solver_choice"] = choice
        save_config(self.cfg)
        # sync bootstrap default
        self.bootstrap_solver_choice.set(choice)
        self.bootstrap_solver_label.set(SOLVER_LABELS[choice])
        self._show_solver_opts()
        self._update_unc_widgets()

    def _on_bootstrap_solver_change(self):
        label = self.bootstrap_solver_label.get()
        choice = SOLVER_LABELS_INV.get(label, "classic")
        self.bootstrap_solver_choice.set(choice)
        self._update_unc_widgets()

    def _update_unc_widgets(self):
        label = SOLVER_LABELS[self.bootstrap_solver_choice.get()]
        self.unc_method_combo["values"] = [
            "Asymptotic",
            f"Bootstrap (base solver = {label})",
            "Bayesian",
        ]
        # Preserve current selection if possible
        current = self.unc_method.get()
        if current.startswith("Bootstrap"):
            self.unc_method.set(f"Bootstrap (base solver = {label})")
        if self.unc_method.get().startswith("Bootstrap"):
            self.bootstrap_solver_combo.pack(side=tk.LEFT, padx=4)
            self.unc_workers_frame.pack(side=tk.LEFT, padx=4)
        else:
            self.bootstrap_solver_combo.pack_forget()
            self.unc_workers_frame.pack_forget()

    def _on_unc_method_change(self, _e=None):
        label = self.unc_method.get()
        if label.startswith("Bootstrap"):
            self.cfg["unc_method"] = "bootstrap"
        elif label.startswith("Bayesian"):
            self.cfg["unc_method"] = "bayesian"
        else:
            self.cfg["unc_method"] = "asymptotic"
        save_config(self.cfg)
        self._update_unc_widgets()

    def _on_unc_workers_change(self, *_):
        self.cfg["unc_workers"] = int(self.unc_workers_var.get())
        save_config(self.cfg)

    def _resolve_unc_workers(self) -> int:
        w = int(self.unc_workers_var.get())
        if w <= 0:
            w = int(self.perf_max_workers.get())
            if w <= 0:
                w = os.cpu_count() or 1
        return w


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
        self.show_ci_band = bool(self.show_ci_band_var.get())
        self.cfg["ui_show_uncertainty_band"] = self.show_ci_band
        save_config(self.cfg)
        self.refresh_plot()

    def _toggle_legend_action(self):
        self.show_legend_var.set(not self.show_legend_var.get())
        self._on_legend_toggle()

    def _fd_jacobian(self, residual, p0):
        p0 = np.asarray(p0, float)
        r0 = residual(p0)
        J = np.empty((r0.size, p0.size), float)
        for j in range(p0.size):
            step = 1e-6 * max(1.0, abs(p0[j]))
            tp = p0.copy()
            tp[j] += step
            J[:, j] = (residual(tp) - r0) / step
        return J, r0

    @staticmethod
    def _svd_cov_from_jacobian(J: np.ndarray, rss: float, m: int) -> tuple[np.ndarray, int, float]:
        """Return PSD covariance via SVD of the Jacobian."""
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        tol = max(J.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        nz = s > tol
        rank = int(np.count_nonzero(nz))
        dof = max(1, m - rank)
        sigma2 = rss / dof
        inv_s2 = np.zeros_like(s)
        inv_s2[nz] = 1.0 / (s[nz] ** 2)
        cov = (Vt.T * inv_s2) @ Vt
        cond = float(s[nz].max() / s[nz].min()) if rank > 0 else np.inf
        return sigma2 * cov, rank, cond

    @staticmethod
    def _safe_sqrt_vec(x: np.ndarray) -> np.ndarray:
        """Clip negatives to zero before sqrt to avoid warnings."""
        return np.sqrt(np.clip(x, 0.0, np.inf))

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

        try:
            self.root.after(0, ui_update)
        except Exception:
            ui_update()

    def log_threadsafe(self, msg: str, level: str = "INFO"):
        self.log(msg, level)

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

    def _label_from_unc(self, obj) -> str:
        # Accept UncertaintyResult-like, SimpleNamespace, or dict
        try:
            lbl = getattr(obj, "label", None) or getattr(obj, "method_label", None)
            if not lbl:
                lbl = getattr(obj, "method", None) or getattr(obj, "type", None)
        except Exception:
            lbl = None
        if not lbl and isinstance(obj, dict):
            lbl = obj.get("label") or obj.get("method") or obj.get("type")
        text = (str(lbl) if lbl else "").lower()

        # Map common aliases
        if "asym" in text or "j" in text and "t" in text and "j" in text:  # covers "jtj", "j^t j"
            return "Asymptotic (JᵀJ)"
        if "boot" in text or "resid" in text:
            return "Bootstrap (residual)"
        if "bayes" in text or "mcmc" in text:
            return "Bayesian (MCMC)"
        # Fallback to original label or generic
        return str(lbl) if lbl else "Unknown"

    def _extract_band(self, obj):
        import numpy as np
        # return (x, lo, hi) or None
        # Attribute forms
        for attr in ("band", "prediction_band"):
            xlh = getattr(obj, attr, None)
            if xlh:
                try:
                    x, lo, hi = xlh
                    x = np.asarray(x); lo = np.asarray(lo); hi = np.asarray(hi)
                    if x.shape == lo.shape == hi.shape and x.size > 0:
                        return (x, lo, hi)
                except Exception:
                    pass
        # Dict forms
        if isinstance(obj, dict):
            # Combined
            if "band" in obj:
                try:
                    x, lo, hi = obj["band"]
                    x = np.asarray(x); lo = np.asarray(lo); hi = np.asarray(hi)
                    if x.shape == lo.shape == hi.shape and x.size > 0:
                        return (x, lo, hi)
                except Exception:
                    pass
            # Split keys
            keys = obj.keys()
            if {"band_x", "band_lo", "band_hi"} <= set(keys):
                try:
                    x = np.asarray(obj["band_x"]); lo = np.asarray(obj["band_lo"]); hi = np.asarray(obj["band_hi"])
                    if x.shape == lo.shape == hi.shape and x.size > 0:
                        return (x, lo, hi)
                except Exception:
                    pass
        return None

    def _unc_extract_stats(self, result):
        """
        Return list of per-peak dicts with keys:
          center_est, center_sd, height_est, height_sd, fwhm_est, fwhm_sd
        Supports:
          - result.stats / parameters / param_stats
          - list-of-dicts rows
          - dict {'center': {'est':[...],'sd':[...]} , ...}
          - flat dict-of-lists {'center_est':[...], 'height_sd':[...], ...}
        """
        import math

        def _pick_idx(container, i, *keys):
            # pick container[key][i] for first existing key; tolerate short arrays
            for k in keys:
                v = container.get(k)
                if v is None:
                    continue
                try:
                    return v[i]
                except Exception:
                    # could be scalar
                    try:
                        return v
                    except Exception:
                        pass
            return None

        def _coerce_row(row):
            # row may be nested: {'center': {'est':..,'sd':..}, 'height': {...}, 'fwhm': {...}}
            out = {}
            # nested blocks
            for pname in ("center", "height", "fwhm"):
                blk = row.get(pname)
                if isinstance(blk, dict):
                    est = blk.get("est") or blk.get("mean") or blk.get("median") or blk.get("value")
                    sd  = blk.get("sd")  or blk.get("std")  or blk.get("se")
                    if est is not None: out[f"{pname}_est"] = est
                    if sd  is not None: out[f"{pname}_sd"]  = sd
            # flat fallbacks
            for pname in ("center", "height", "fwhm"):
                for k in (f"{pname}_est", pname):
                    if k in row and row[k] is not None:
                        out.setdefault(f"{pname}_est", row[k])
                        break
                for k in (f"{pname}_sd", f"{pname}_std", f"{pname}_se"):
                    if k in row and row[k] is not None:
                        out.setdefault(f"{pname}_sd", row[k])
                        break
            return out

        # locate stats object
        stats = None
        for attr in ("stats", "parameters", "param_stats"):
            if hasattr(result, attr):
                stats = getattr(result, attr)
                break
        if stats is None and isinstance(result, dict):
            for k in ("stats", "parameters", "param_stats"):
                if k in result:
                    stats = result[k]
                    break
        if stats is None:
            return None

        # case A: list-of-dicts
        if isinstance(stats, (list, tuple)):
            rows = []
            for row in stats:
                if isinstance(row, dict):
                    rows.append(_coerce_row(row))
            return rows if rows else None

        # case B: dict form
        if isinstance(stats, dict):
            # B1) block-of-arrays: {'center': {'est':[...],'sd':[...]} , ...}
            if all(k in stats for k in ("center", "height", "fwhm")) and \
               all(isinstance(stats[k], dict) for k in ("center", "height", "fwhm")):
                n = 0
                for pname in ("center", "height", "fwhm"):
                    for key in ("est", "mean", "median", "value"):
                        v = stats[pname].get(key)
                        if hasattr(v, "__len__"):
                            n = max(n, len(v))
                rows = []
                for i in range(n):
                    row = {
                        "center": {"est": _pick_idx(stats["center"], i, "est", "mean", "median", "value"),
                                   "sd":  _pick_idx(stats["center"], i, "sd", "std", "se")},
                        "height": {"est": _pick_idx(stats["height"], i, "est", "mean", "median", "value"),
                                   "sd":  _pick_idx(stats["height"], i, "sd", "std", "se")},
                        "fwhm":   {"est": _pick_idx(stats["fwhm"],   i, "est", "mean", "median", "value"),
                                   "sd":  _pick_idx(stats["fwhm"],   i, "sd", "std", "se")},
                    }
                    rows.append(_coerce_row(row))
                return rows
            # B2) flat dict-of-lists: {'center_est':[...], 'height_sd':[...], ...}
            # compute max length
            n = 0
            for v in stats.values():
                if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                    n = max(n, len(v))
            if n == 0:
                return None
            rows = []
            for i in range(n):
                row = {}
                for k, v in stats.items():
                    try:
                        row[k] = v[i] if hasattr(v, "__getitem__") else v
                    except Exception:
                        row[k] = v
                rows.append(_coerce_row(row))
            return rows

        return None

    def run_in_thread(self, fn, on_done):
        def worker():
            try:
                res = fn()
                err = None
            except Exception as e:
                res, err = None, e
            def _cb():
                on_done(res, err)
            self.root.after(0, _cb)
        threading.Thread(target=worker, daemon=True).start()

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
        lam = float(self.als_lam.get())
        asym = float(self.als_asym.get())
        niter = int(self.als_niter.get())
        thresh = float(self.als_thresh.get())
        use_slice = bool(self.baseline_use_range.get())
        mask = self._range_mask() if use_slice else None

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
                    base = signals.als_baseline(self.y_raw, lam=lam, p=asym,
                                                niter=niter, tol=thresh)
                else:
                    x_sub = self.x[mask]
                    y_sub = self.y_raw[mask]
                    z_sub = signals.als_baseline(y_sub, lam=lam, p=asym,
                                                 niter=niter, tol=thresh)
                    base = np.interp(self.x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
                self.baseline = base
                if key is not None:
                    self._baseline_cache[key] = base
            except Exception as e:
                messagebox.showwarning("Baseline", f"ALS baseline failed: {e}")
                self.baseline = np.zeros_like(self.y_raw)

        try:
            y_t = self.get_fit_target()
            snr = signals.snr_estimate(y_t if y_t is not None else self.y_raw)
            self.snr_text.set(f"S/N: {snr:.2f}")
        except Exception:
            self.snr_text.set("S/N: --")

        self.refresh_plot()

    def save_baseline_default(self):
        self.cfg["als_lam"] = float(self.als_lam.get())
        self.cfg["als_asym"] = float(self.als_asym.get())
        self.cfg["als_niter"] = int(self.als_niter.get())
        self.cfg["als_thresh"] = float(self.als_thresh.get())
        save_config(self.cfg)
        messagebox.showinfo("Baseline", "Saved as default for future sessions.")

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
             "eta": float(p.eta), "lock_width": bool(p.lock_width), "lock_center": bool(p.lock_center)}
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
                            lock_center=bool(spk.get("lock_center", False))))
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

    def on_lock_toggle(self):
        sel = self._selected_index()
        if sel is None:
            return
        pk = self.peaks[sel]
        pk.lock_width  = bool(self.lockw_var.get())
        pk.lock_center = bool(self.lockc_var.get())
        self.refresh_tree(keep_selection=True)
        self.refresh_plot()

    def add_peak_from_fields(self):
        try:
            c = float(self.center_var.get())
            h = float(self.height_var.get())
            w = max(float(self.fwhm_var.get()), 1e-6)
        except ValueError:
            messagebox.showwarning("Add Peak", "Center, Height, and FWHM must be numbers.")
            return
        pk = Peak(center=c, height=h, fwhm=w, eta=float(self.global_eta.get()),
                  lock_center=bool(self.lockc_var.get()), lock_width=bool(self.lockw_var.get()))
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

        y_target = self.get_fit_target()

        mask = self.current_fit_mask()
        if mask is None or not np.any(mask):
            messagebox.showwarning("Fit", "Fit range is empty. Use 'Full range' or set a valid Min/Max.")
            return
        x_fit = self.x[mask]
        y_fit = y_target[mask]

        base_applied = self.use_baseline.get() and self.baseline is not None
        add_mode = (self.baseline_mode.get() == "add")
        base_fit = self.baseline[mask] if (base_applied and add_mode) else None
        mode = "add" if add_mode else "subtract"

        solver = self.solver_choice.get()
        options = self._solver_options()
        options["solver"] = solver

        def work():
            return orchestrator.run_fit_with_fallbacks(
                x_fit, y_fit, self.peaks, mode, base_fit, options
            )

        def done(res, err):
            self.step_btn.config(state=tk.NORMAL)
            if err or res is None:
                self.set_busy(False, "Fit failed.")
                if err:
                    self.log(traceback.format_exception_only(type(err), err)[0].strip(), level="ERROR")
                    messagebox.showerror("Fit", f"Fitting failed:\n{err}")
                return
            self.peaks[:] = res.peaks_out
            self.refresh_tree(keep_selection=True)
            self.last_unc_result = None
            self.last_unc_method = None
            self.ci_band = None
            self.show_ci_band = False
            self.refresh_plot()
            self.set_busy(False, f"Fit done. RMSE {res.rmse:.4g}")
            npts = int(np.count_nonzero(mask))
            self.log(
                f"Fit finished: RMSE={res.rmse:.4g} over {npts} pts (peaks={len(self.peaks)})"
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
        p = Path(out_csv)
        base = p.with_suffix("")
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

        solver = self.solver_choice.get()
        cfg = {
            "peaks": peaks_list,
            "solver": solver,
            "mode": self.baseline_mode.get(),
            "baseline": {
                "lam": float(self.als_lam.get()),
                "p": float(self.als_asym.get()),
                "niter": int(self.als_niter.get()),
                "thresh": float(self.als_thresh.get()),
            },
            "save_traces": bool(self.batch_save_traces.get()),
            "source": source,
            "reheight": bool(self.batch_reheight.get()),
            "auto_max": int(self.batch_auto_max.get()),
            solver: self._solver_options(solver),
            "baseline_uses_fit_range": bool(self.baseline_use_range.get()),
            "perf_numba": bool(self.perf_numba.get()),
            "perf_gpu": bool(self.perf_gpu.get()),
            "perf_cache_baseline": bool(self.perf_cache_baseline.get()),
            "perf_seed_all": bool(self.perf_seed_all.get()),
            "perf_max_workers": int(self.perf_max_workers.get()),
            "unc_workers": int(self.cfg.get("unc_workers", 0)),
            "output_dir": str(output_dir),
            "output_base": output_base,
        }

        if self.fit_xmin is not None and self.fit_xmax is not None:
            cfg["fit_xmin"] = float(self.fit_xmin)
            cfg["fit_xmax"] = float(self.fit_xmax)

        self.cfg["batch_patterns"] = self.batch_patterns.get()
        self.cfg["batch_source"] = source
        self.cfg["batch_reheight"] = bool(self.batch_reheight.get())
        self.cfg["batch_auto_max"] = int(self.batch_auto_max.get())
        self.cfg["batch_save_traces"] = bool(self.batch_save_traces.get())
        self.cfg["batch_compute_uncertainty"] = bool(self.batch_unc_enabled.get())
        save_config(self.cfg)

        compute_unc = bool(self.batch_unc_enabled.get())
        unc_method = str(self.unc_method.get()).strip()

        def work():
            def prog(i, total, path):
                self.root.after(0, lambda: self.status_var.set(f"Batch {i}/{total}: {Path(path).name}"))

            return batch_runner.run_batch(
                patterns,
                cfg,
                compute_uncertainty=compute_unc,
                unc_method=unc_method,
                progress=prog,
                log=self.log_threadsafe,
            )

        def done(res, err):
            if err or res is None:
                self.set_busy(False, "Batch failed.")
                if err:
                    self.log(f"Batch failed: {err}", level="ERROR")
                    messagebox.showerror("Batch", f"Batch failed:\n{err}")
                return
            ok, total = res
            self.set_busy(False, f"Batch done. {ok}/{total} succeeded.")
            self.log(f"Batch done: {ok}/{total} succeeded.")
            messagebox.showinfo("Batch", f"Summary saved:\n{fit_path}")

        self.set_busy(True, "Batch running…")
        self.run_in_thread(work, done)

    def _run_asymptotic_uncertainty(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            self.ci_band = None
            return None
        mask = self.current_fit_mask()
        if mask is None or not np.any(mask):
            self.ci_band = None
            return None
        x_fit = self.x[mask]
        y_fit = self.get_fit_target()[mask]
        base_applied = self.use_baseline.get() and self.baseline is not None
        add_mode = (self.baseline_mode.get() == "add")
        base_fit = self.baseline[mask] if (base_applied and add_mode) else None
        mode = "add" if add_mode else "subtract"

        theta = []
        for p in self.peaks:
            theta.extend([p.center, p.height, p.fwhm, p.eta])
        theta = np.asarray(theta, float)

        resid_fn = build_residual(x_fit, y_fit, self.peaks, mode, base_fit, "linear", None)
        J, r0 = self._fd_jacobian(resid_fn, theta)
        rss = float(np.dot(r0, r0))
        m = r0.size
        cov, rank, cond = self._svd_cov_from_jacobian(J, rss, m)
        self.param_sigma = self._safe_sqrt_vec(np.diag(cov))

        x_all = self.x
        base_all = self.baseline if (base_applied and add_mode) else None

        def ymodel(th):
            total = np.zeros_like(x_all)
            for i in range(len(self.peaks)):
                c, h, fw, eta = th[4 * i : 4 * i + 4]
                total += pseudo_voigt(x_all, h, c, fw, eta)
            if base_all is not None:
                total = total + base_all
            return total

        y0 = ymodel(theta)
        G = np.empty((x_all.size, theta.size), float)
        for j in range(theta.size):
            step = 1e-6 * max(1.0, abs(theta[j]))
            tp = theta.copy()
            tp[j] += step
            G[:, j] = (ymodel(tp) - y0) / step

        var = np.einsum('ij,jk,ik->i', G, cov, G)
        band_std = self._safe_sqrt_vec(var)
        z = 1.96
        lo = y0 - z * band_std
        hi = y0 + z * band_std
        warn_nonfinite = False
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            lo = np.nan_to_num(lo)
            hi = np.nan_to_num(hi)
            warn_nonfinite = True
        self.ci_band = (x_all, lo, hi, y0)
        self.show_ci_band = True

        bw = (hi - lo)[mask]
        bw_stats = (float(np.min(bw)), float(np.median(bw)), float(np.max(bw)))

        dof = max(m - rank, 1)
        info = {
            "m": m,
            "n": theta.size,
            "rank": rank,
            "dof": dof,
            "cond": cond,
            "rmse": math.sqrt(rss / m),
            "s2": rss / dof,
            "bw": bw_stats,
            "warn_nonfinite": warn_nonfinite,
        }
        self.unc_info = info
        return cov, theta, info

    def _format_asymptotic_summary(self, cov, theta, info, band):
        lines: list[str] = []
        lines.append(
            f"Uncertainty (asymptotic): m={info['m']}, n={info['n']}, rank={info['rank']}, dof={info['dof']}, cond={info['cond']:.3g}"
        )
        bw_min, bw_med, bw_max = info["bw"]
        lines.append(
            f"Band width (95% CI): min={bw_min:.3g}, median={bw_med:.3g}, max={bw_max:.3g}"
        )
        warns: list[str] = []
        if info['rank'] < info['n']:
            warns.append("rank-deficient Jacobian; intervals may be wide.")
        if info.get('cond', np.inf) > 1e8:
            warns.append(
                f"Ill-conditioning detected: cond(JᵀJ)={info['cond']:.3g}; some σ may be inflated (consider locking or narrowing range)"
            )
        if info.get('warn_nonfinite'):
            warns.append("CI band contained non-finite values; replaced with zeros.")
        std = self._safe_sqrt_vec(np.diag(cov))
        for i, _p in enumerate(self.peaks, 1):
            idx = 4 * (i - 1)
            c, h, w = theta[idx: idx + 3]
            sc, sh, sw = std[idx: idx + 3]
            lines.append(
                f"Peak {i} @ center={c:.5g} ± {sc:.3g}; height={h:.3g} ± {sh:.3g}; FWHM={w:.3g} ± {sw:.3g} (CI95 ≈ ±1.96σ)"
            )
        if np.any(~np.isfinite(std)):
            warns.append("Uncertain parameter(s) due to poor conditioning; try adjusting locks, range, or η.")
        lines.append(
            "Tip: σ is the standard error of the fitted parameter. CI95 ≈ ±1.96σ assumes local linearity near the solution."
        )
        return lines, warns

    def run_uncertainty(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            messagebox.showinfo("Uncertainty", "Load data and perform a fit first.")
            return
        self._sync_selected_edits()
        mask = self.current_fit_mask()
        if mask is None or not np.any(mask):
            messagebox.showwarning("Uncertainty", "Fit range is empty. Use 'Full range' or set a valid Min/Max.")
            return
        x_fit = self.x[mask]
        y_fit = self.get_fit_target()[mask]
        base_applied = self.use_baseline.get() and self.baseline is not None
        add_mode = (self.baseline_mode.get() == "add")
        base_fit = self.baseline[mask] if (base_applied and add_mode) else None
        mode = "add" if add_mode else "subtract"

        theta = []
        for p in self.peaks:
            theta.extend([p.center, p.height, p.fwhm, p.eta])
        theta = np.asarray(theta, dtype=float)

        resid_fn = build_residual(x_fit, y_fit, self.peaks, mode, base_fit, "linear", None)

        method_label = self.unc_method.get()
        method = "bootstrap" if method_label.startswith("Bootstrap") else method_label.lower()

        def work():
            if method == "asymptotic":
                res = self._run_asymptotic_uncertainty()
                if res is None:
                    return None
                cov, th, _info = res
                sigma = self._safe_sqrt_vec(np.diag(np.asarray(cov, float)))
                param_stats = {
                    "center": {"est": [p.center for p in self.peaks], "sd": sigma[0::4]},
                    "fwhm": {"est": [p.fwhm for p in self.peaks], "sd": sigma[2::4]},
                    "height": {"est": [p.height for p in self.peaks], "sd": sigma[1::4]},
                }
                return {
                    "method": "asymptotic",
                    "method_label": "Asymptotic (JᵀJ)",
                    "band": self.ci_band[:3] if self.ci_band else None,
                    "param_stats": param_stats,
                }
            if method == "bootstrap":
                cfg = {
                    "x": x_fit,
                    "y": y_fit,
                    "peaks": self.peaks,
                    "mode": mode,
                    "baseline": base_fit,
                    "theta": theta,
                    "options": self._solver_options(self.bootstrap_solver_choice.get()),
                    "n": 100,
                    "workers": self._resolve_unc_workers(),
                }
                res = bootstrap.bootstrap(self.bootstrap_solver_choice.get(), cfg, resid_fn)
                if isinstance(res, dict):
                    cb = res.get("curve_band") or {}
                    if isinstance(cb, dict) and {"x", "lo", "hi"} <= set(cb.keys()):
                        res["band"] = (cb.get("x"), cb.get("lo"), cb.get("hi"))
                    if "params" in res and "param_stats" not in res:
                        try:
                            params = res.get("params", {})
                            th = np.asarray(params.get("theta", []), float)
                            cov = params.get("cov")
                            sd = (
                                np.sqrt(np.diag(np.asarray(cov, float)))
                                if cov is not None and np.size(cov) > 0
                                else None
                            )
                            samples = params.get("samples")
                            p_lo = p_hi = None
                            if samples is not None and np.size(samples) > 0:
                                samp = np.asarray(samples, float)
                                p_lo = np.quantile(samp, 0.025, axis=0)
                                p_hi = np.quantile(samp, 0.975, axis=0)

                            def slice_stats(idx: int) -> Dict[str, Any]:
                                est = th[idx::4] if th.size else None
                                sd_i = sd[idx::4] if sd is not None else None
                                d: Dict[str, Any] = {"est": est, "sd": sd_i}
                                if p_lo is not None and p_hi is not None:
                                    d["p2_5"] = p_lo[idx::4]
                                    d["p97_5"] = p_hi[idx::4]
                                return d

                            res["param_stats"] = {
                                "center": slice_stats(0),
                                "height": slice_stats(1),
                                "fwhm": slice_stats(2),
                            }
                        except Exception:
                            pass
                return res
            if method == "bayesian":
                init = {"x": x_fit, "y": y_fit, "peaks": self.peaks, "mode": mode,
                        "baseline": base_fit, "theta": theta}
                res = bayes.bayesian({}, "gaussian", init, {}, resid_fn)
                if isinstance(res, dict):
                    cb = res.get("curve_band") or {}
                    if isinstance(cb, dict) and {"x", "lo", "hi"} <= set(cb.keys()):
                        res["band"] = (cb.get("x"), cb.get("lo"), cb.get("hi"))
                    if "params" in res and "param_stats" not in res:
                        try:
                            params = res.get("params", {})
                            th = np.asarray(params.get("theta", []), float)
                            cov = params.get("cov")
                            sd = (
                                np.sqrt(np.diag(np.asarray(cov, float)))
                                if cov is not None and np.size(cov) > 0
                                else None
                            )
                            samples = params.get("samples")
                            p_lo = p_hi = None
                            if samples is not None and np.size(samples) > 0:
                                samp = np.asarray(samples, float)
                                p_lo = np.quantile(samp, 0.025, axis=0)
                                p_hi = np.quantile(samp, 0.975, axis=0)

                            def slice_stats(idx: int) -> Dict[str, Any]:
                                est = th[idx::4] if th.size else None
                                sd_i = sd[idx::4] if sd is not None else None
                                d: Dict[str, Any] = {"est": est, "sd": sd_i}
                                if p_lo is not None and p_hi is not None:
                                    d["p2_5"] = p_lo[idx::4]
                                    d["p97_5"] = p_hi[idx::4]
                                return d

                            res["param_stats"] = {
                                "center": slice_stats(0),
                                "height": slice_stats(1),
                                "fwhm": slice_stats(2),
                            }
                        except Exception:
                            pass
                return res
            raise RuntimeError("Unknown method")

        def done(result, error):
            # Drop stale callbacks (auto asymptotic finishing after a newer user-triggered run)
            if job_id != getattr(self, "_unc_job_id", 0):
                self._progress_end("uncertainty")
                return

            self._unc_running = False
            self._progress_end("uncertainty")

            if error is not None:
                self.status_error(f"Uncertainty failed: {error}")
                return

            label = self._label_from_unc(result)
            self.last_unc_result = result
            self.last_unc_method = label

            # De-dupe “Computed …” for this job+label
            if getattr(self, "_last_unc_log", None) == (job_id, label):
                return
            self._last_unc_log = (job_id, label)

            # Band
            band = self._extract_band(result)
            self.status_info(f"Computed {label} uncertainty.")
            if band is not None:
                x, lo, hi = band[:3]
                try:
                    self.ci_band = (np.asarray(x), np.asarray(lo), np.asarray(hi))
                except Exception:
                    self.ci_band = band
                self.show_ci_band = True
                try:
                    self.refresh_plot()
                except Exception:
                    pass
            reason = getattr(result, "diagnostics", {}).get("band_reason")
            if band is None and reason:
                self.status_warn(f"{label}: no band — {reason}")

            # Per-peak stats
            try:
                rows = self._unc_extract_stats(result)
                if rows:
                    for i, row in enumerate(rows, 1):
                        def _fmt(v_est, v_sd):
                            try:
                                s_est = f"{float(v_est):.6g}" if v_est is not None else "n/a"
                            except Exception:
                                s_est = "n/a"
                            try:
                                s_sd = f"{float(v_sd):.3g}" if v_sd is not None else "n/a"
                            except Exception:
                                s_sd = "n/a"
                            return s_est, s_sd
                        c_est, c_sd = _fmt(row.get("center_est"), row.get("center_sd"))
                        h_est, h_sd = _fmt(row.get("height_est"), row.get("height_sd"))
                        w_est, w_sd = _fmt(row.get("fwhm_est"),  row.get("fwhm_sd"))
                        self.status_info(
                            f"Peak {i}: center={c_est} ± {c_sd} | height={h_est} ± {h_sd} | FWHM={w_est} ± {w_sd}"
                        )
            except Exception as _e:
                self.status_warn(f"Uncertainty stats formatting skipped ({_e.__class__.__name__}).")

        self._unc_running = True
        self._unc_job_id += 1
        job_id = self._unc_job_id
        self._progress_begin("uncertainty")
        self.status_info("Computing uncertainty…")
        self.run_in_thread(work, done)

    def apply_performance(self):
        performance.set_numba(bool(self.perf_numba.get()))
        performance.set_gpu(bool(self.perf_gpu.get()))
        performance.set_cache_baseline(bool(self.perf_cache_baseline.get()))
        seed_txt = self.seed_var.get().strip()
        seed = int(seed_txt) if seed_txt else None
        if self.perf_seed_all.get():
            performance.set_seed(seed)
        else:
            performance.set_seed(None)
        performance.set_max_workers(int(self.perf_max_workers.get()))
        performance.set_gpu_chunk(self.gpu_chunk_var.get())
        self.cfg["perf_numba"] = bool(self.perf_numba.get())
        self.cfg["perf_gpu"] = bool(self.perf_gpu.get())
        self.cfg["perf_cache_baseline"] = bool(self.perf_cache_baseline.get())
        self.cfg["perf_seed_all"] = bool(self.perf_seed_all.get())
        self.cfg["perf_max_workers"] = int(self.perf_max_workers.get())
        save_config(self.cfg)
        self.log(f"Backend: {performance.which_backend()} | workers={performance.get_max_workers()}")
        self.status_var.set("Performance options applied.")

    def _maybe_export_uncertainty(
        self, txt_path: Path, csv_path: Path, band_path: Path, rmse: float
    ) -> None:
        try:
            if self.ci_band is None or getattr(self, "param_sigma", None) is None:
                try:
                    self._run_asymptotic_uncertainty()
                except Exception as e:
                    self.log(f"Uncertainty failed: {e}", level="WARN")

            opts = self._solver_options()
            solver = self.solver_choice.get()
            fname_disp = self.current_file.name if self.current_file else "(unsaved)"
            lines = [f"File: {fname_disp}", "Uncertainty method: Asymptotic (95% CI, z=1.96)"]
            lines.append(
                "Solver: "
                f"{solver}, loss={opts.get('loss', '')}, weight={opts.get('weights', '')}, "
                f"f_scale={opts.get('f_scale', '')}, maxfev={opts.get('maxfev', '')}, "
                f"restarts={opts.get('restarts', '')}, jitter_pct={opts.get('jitter_pct', '')}"
            )
            lines.append(
                "Baseline: "
                f"uses_fit_range={bool(self.baseline_use_range.get())}, "
                f"lam={self.als_lam.get()}, p={self.als_asym.get()}, "
                f"niter={self.als_niter.get()}, thresh={self.als_thresh.get()}"
            )
            lines.append(
                "Performance: "
                f"numba={bool(self.perf_numba.get())}, gpu={bool(self.perf_gpu.get())}, "
                f"cache_baseline={bool(self.perf_cache_baseline.get())}, "
                f"seed_all={bool(self.perf_seed_all.get())}, max_workers={int(self.perf_max_workers.get())}"
            )
            lines.append("Peaks:")
            z = 1.96
            sigma = getattr(self, "param_sigma", np.array([]))
            dof = getattr(self, "unc_info", {}).get("dof", np.nan)
            rows = []
            fname = self.current_file.name if self.current_file else ""
            for i, p in enumerate(self.peaks, 1):
                lines.append(f"Peak {i}")
                sc = sigma[4 * (i - 1)] if sigma.size >= 4 * i else np.nan
                sh = sigma[4 * (i - 1) + 1] if sigma.size >= 4 * i + 1 else np.nan
                sf = sigma[4 * (i - 1) + 2] if sigma.size >= 4 * i + 2 else np.nan
                se = sigma[4 * (i - 1) + 3] if sigma.size >= 4 * i + 3 else np.nan
                params = [
                    ("center", p.center, sc, not p.lock_center),
                    ("height", p.height, sh, True),
                    ("fwhm", p.fwhm, sf, not p.lock_width),
                    ("eta", p.eta, se, True),
                ]
                for pname, val, std, free in params:
                    if free and np.isfinite(std):
                        ci_lo = val - z * std
                        ci_hi = val + z * std
                        lines.append(
                            f"  {pname:<7}= {val:.6g} ± {std:.3g}   (95% CI: [{ci_lo:.6g}, {ci_hi:.6g}])"
                        )
                        rows.append(
                            {
                                "file": fname,
                                "peak": i,
                                "param": pname,
                                "value": val,
                                "stderr": std,
                                "ci_lo": ci_lo,
                                "ci_hi": ci_hi,
                                "method": "asymptotic",
                                "rmse": rmse,
                                "dof": dof,
                            }
                        )
                    else:
                        lines.append(f"  {pname:<7}= {val:.6g} (fixed)")
                        rows.append(
                            {
                                "file": fname,
                                "peak": i,
                                "param": pname,
                                "value": val,
                                "stderr": np.nan,
                                "ci_lo": np.nan,
                                "ci_hi": np.nan,
                                "method": "asymptotic",
                                "rmse": rmse,
                                "dof": dof,
                            }
                        )

            lines.append(f"Fit quality: RMSE={rmse:.5g}, DOF={dof}")
            with txt_path.open("w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join(lines) + "\n")

            header = [
                "file",
                "peak",
                "param",
                "value",
                "stderr",
                "ci_lo",
                "ci_hi",
                "method",
                "rmse",
                "dof",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            if self.ci_band is not None:
                xb, lob, hib, yfit = self.ci_band
                with band_path.open("w", newline="", encoding="utf-8") as fh:
                    bw = csv.writer(fh, lineterminator="\n")
                    bw.writerow(["x", "y_fit", "y_lo95", "y_hi95"])
                    for xi, yi, lo, hi in zip(xb, yfit, lob, hib):
                        bw.writerow([xi, yi, lo, hi])
        except Exception as e:  # pragma: no cover - defensive
            self.log(f"Uncertainty export failed: {e}", level="WARN")
            self.status_var.set("Uncertainty export failed.")

    def on_export(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            messagebox.showinfo("Export", "Load data and perform a fit first.")
            return
        out_csv = filedialog.asksaveasfilename(
            title="Save export base name",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out_csv:
            return
        paths = data_io.derive_export_paths(out_csv)

        areas = [pseudo_voigt_area(p.height, p.fwhm, p.eta) for p in self.peaks]
        total_area = float(np.sum(areas)) if areas else 1.0

        y_target = self.get_fit_target()
        total_peaks = np.zeros_like(self.x, float)
        comp_cols = {}
        for i, p in enumerate(self.peaks, 1):
            comp = pseudo_voigt(self.x, p.height, p.center, p.fwhm, p.eta)
            comp_cols[f"peak{i}"] = comp
            total_peaks += comp

        base = self.baseline if (self.use_baseline.get() and self.baseline is not None) else np.zeros_like(self.x)
        if self.use_baseline.get() and self.baseline_mode.get() == "add":
            y_fit = base + total_peaks
            y_corr = self.y_raw - base  # for reference
        else:
            y_fit = total_peaks
            y_corr = self.y_raw - base if self.use_baseline.get() else self.y_raw

        mask = self.current_fit_mask()
        rmse = float(np.sqrt(np.mean((y_target[mask] - y_fit[mask]) ** 2))) if mask is not None else float("nan")

        rows = []
        fname = self.current_file.name if self.current_file else ""
        opts = self._solver_options()
        solver = self.solver_choice.get()
        perf_extras = {
            "perf_numba": bool(self.perf_numba.get()),
            "perf_gpu": bool(self.perf_gpu.get()),
            "perf_cache_baseline": bool(self.perf_cache_baseline.get()),
            "perf_seed_all": bool(self.perf_seed_all.get()),
            "perf_max_workers": int(self.perf_max_workers.get()),
        }
        center_bounds = (self.fit_xmin, self.fit_xmax) if (opts.get("centers_in_window") or opts.get("bound_centers_to_window")) else (np.nan, np.nan)
        if self.x is not None and self.x.size > 1:
            med_dx = float(np.median(np.diff(np.sort(self.x))))
        else:
            med_dx = 0.0
        fwhm_lo = opts.get("min_fwhm", max(1e-6, 2.0 * med_dx))
        for i, (p, a) in enumerate(zip(self.peaks, areas), 1):
            row = {
                "file": fname,
                "peak": i,
                "center": p.center,
                "height": p.height,
                "fwhm": p.fwhm,
                "eta": p.eta,
                "lock_width": p.lock_width,
                "lock_center": p.lock_center,
                "area": a,
                "area_pct": 100.0 * a / total_area,
                "rmse": rmse,
                "fit_ok": True,
                "mode": self.baseline_mode.get(),
                "als_lam": float(self.als_lam.get()),
                "als_p": float(self.als_asym.get()),
                "fit_xmin": self.fit_xmin if self.fit_xmin is not None else float(self.x.min()),
                "fit_xmax": self.fit_xmax if self.fit_xmax is not None else float(self.x.max()),
                "solver_choice": solver,
                "solver_loss": opts.get("loss", np.nan),
                "solver_weight": opts.get("weights", np.nan),
                "solver_fscale": opts.get("f_scale", np.nan),
                "solver_maxfev": opts.get("maxfev", np.nan),
                "solver_restarts": opts.get("restarts", np.nan),
                "solver_jitter_pct": opts.get("jitter_pct", np.nan),
                "use_baseline": bool(self.use_baseline.get()),
                "baseline_mode": self.baseline_mode.get(),
                "baseline_uses_fit_range": bool(self.baseline_use_range.get()),
                "als_niter": int(self.als_niter.get()),
                "als_thresh": float(self.als_thresh.get()),
                **perf_extras,
                "bounds_center_lo": center_bounds[0],
                "bounds_center_hi": center_bounds[1],
                "bounds_fwhm_lo": fwhm_lo,
                "bounds_height_lo": 0.0,
                "bounds_height_hi": np.nan,
                "x_scale": opts.get("x_scale", np.nan),
            }
            rows.append(row)
        peak_csv = data_io.build_peak_table(rows)
        with open(paths["fit"], "w", encoding="utf-8", newline="") as fh:
            fh.write(peak_csv)

        trace_csv = data_io.build_trace_table(
            self.x, self.y_raw, base if self.use_baseline.get() else None, self.peaks
        )
        with open(paths["trace"], "w", encoding="utf-8", newline="") as fh:
            fh.write(trace_csv)

        saved = [paths["fit"], paths["trace"]]
        if self.last_unc_result is not None:
            try:
                data_io.write_uncertainty_txt(
                    paths["unc_txt"],
                    self.last_unc_result,
                    peaks=self.peaks,
                    method_label=self.last_unc_method or ""
                )
                data_io.write_uncertainty_csv(
                    paths["unc_csv"],
                    self.last_unc_result,
                    peaks=self.peaks,
                    method_label=self.last_unc_method or ""
                )
                band = data_io._normalize_band(self.last_unc_result)
                if band is not None:
                    xb, lob, hib = band
                    with open(paths["unc_band"], "w", newline="", encoding="utf-8") as fh:
                        bw = csv.writer(fh, lineterminator="\n")
                        bw.writerow(["x", "y_lo95", "y_hi95"])
                        for xi, lo, hi in zip(xb, lob, hib):
                            bw.writerow([xi, lo, hi])
                    saved.append(paths["unc_band"])
                saved.extend([paths["unc_txt"], paths["unc_csv"]])
                self.status_info(f"Exported uncertainty ({self.last_unc_method}).")
            except Exception as e:  # pragma: no cover - defensive
                self.log(f"Uncertainty export failed: {e}", level="WARN")
        else:
            self.status_info("No uncertainty computed — skipping uncertainty export.")

        saved_lines = [str(p) for p in saved if p]
        messagebox.showinfo("Export", "Saved:\n" + "\n".join(saved_lines))

    # ----- Plot -----
    def toggle_components(self):
        self.components_visible = not self.components_visible
        self.cfg["ui_show_components"] = self.components_visible
        save_config(self.cfg)
        self.refresh_plot()

    def refresh_plot(self):
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

        if self.show_ci_band and self.ci_band is not None:
            try:
                if len(self.ci_band) == 4:
                    x_b, lo_b, hi_b, _ = self.ci_band
                else:
                    x_b, lo_b, hi_b = self.ci_band
                self.ax.fill_between(
                    np.asarray(x_b), np.asarray(lo_b), np.asarray(hi_b),
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


def main():
    root = tk.Tk()
    app = PeakFitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
