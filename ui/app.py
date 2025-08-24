#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive peak fit GUI for spectra (Gaussian–Lorentzian / pseudo-Voigt)
Designed by Farhan Zahin
Built with ChatGPT

Build: v2.7+

Features:
  • ALS baseline with saved defaults
  • Baseline modes: "Add to fit" (baseline + peaks) or "Subtract" (peaks on y - baseline)
  • Option: compute ALS baseline only within a chosen x-range, then interpolate across full x
  • Iteration/threshold controls with S/N readout
  • Thin plot lines; components drawn on top of baseline in "Add to fit" mode
  • Click to add peaks (toggleable); ignores clicks while Zoom/Pan is active
  • Lock width and/or center per-peak (applies instantly)
  • Global η (Gaussian–Lorentzian shape factor) with "Apply to all"
  • Auto-seed peaks (respects fit range)
  • Choose a fit x-range (type Min/Max or drag with a SpanSelector); shaded on plot
  • Solver selection (Classic, Modern, LMFIT) plus Step ▶ single iteration
  • Multiple peak templates (save as new, save changes, select/apply, delete); optional auto-apply on open
  • Zoom out & Reset view buttons
  • Supports CSV, TXT, DAT (auto delimiter detection; skips headers/comments)
  • Export peak table with metadata and full trace CSV (raw, baseline, components)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from scipy.signal import find_peaks

from core import signals
from core.residuals import build_residual
from fit import classic, lmfit_backend, modern, step_engine
from infra import performance
from uncertainty import asymptotic, bayes, bootstrap


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
    "auto_apply_template_name": ""
}

def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            cfg = {**DEFAULTS, **data}
            # Migration: move legacy saved_peaks into templates["default"] if templates is empty
            if cfg.get("saved_peaks") and not cfg.get("templates"):
                cfg["templates"] = {"default": cfg["saved_peaks"]}
            return cfg
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        messagebox.showwarning("Config", f"Could not save config: {e}")


# ---------- Fitting utilities ----------
# ---------- File loader (CSV/TXT/DAT) ----------
def load_xy_any(path: str):
    """Wrapper around :func:`core.data_io.load_xy` for backwards compatibility."""

    from core import data_io

    return data_io.load_xy(path)


# ---------- Main GUI ----------
class PeakFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Origin-like Peak Fit (pseudo-Voigt)")

        # Data
        self.x = None
        self.y_raw = None
        self.baseline = None
        self.use_baseline = tk.BooleanVar(value=True)

        # Baseline mode: "add" (fit over baseline) or "subtract"
        self.baseline_mode = tk.StringVar(value="add")
        # Option: compute ALS baseline only from fit range
        self.baseline_use_range = tk.BooleanVar(value=False)

        # Config
        self.cfg = load_config()
        self.als_lam = tk.DoubleVar(value=self.cfg["als_lam"])
        self.als_asym = tk.DoubleVar(value=self.cfg["als_asym"])
        self.als_niter = tk.IntVar(value=self.cfg["als_niter"])
        self.als_thresh = tk.DoubleVar(value=self.cfg["als_thresh"])
        self.global_eta = tk.DoubleVar(value=0.5)
        self.auto_apply_template = tk.BooleanVar(value=bool(self.cfg.get("auto_apply_template", False)))
        self.auto_apply_template_name = tk.StringVar(value=self.cfg.get("auto_apply_template_name", ""))

        # Interaction
        self.add_peaks_mode = tk.BooleanVar(value=True)  # click-to-add toggle

        # Fit range (None = full)
        self.fit_xmin: Optional[float] = None
        self.fit_xmax: Optional[float] = None
        self.fit_min_var = tk.StringVar(value="")
        self.fit_max_var = tk.StringVar(value="")
        self.span: Optional[SpanSelector] = None

        # Peaks
        self.peaks: List[Peak] = []

        # Templates UI state
        self.template_var = tk.StringVar(value=self.auto_apply_template_name.get())

        # Components visibility
        self.components_visible = True

        # Solver selection and diagnostics
        self.solver_var = tk.StringVar(value="Classic")
        self.snr_text = tk.StringVar(value="S/N: --")

        # Uncertainty and performance controls
        self.unc_method = tk.StringVar(value="Asymptotic")
        self.perf_numba = tk.BooleanVar(value=False)
        self.perf_gpu = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="")
        self.workers_var = tk.IntVar(value=0)

        # UI
        self._build_ui()
        self._new_figure()
        self._update_template_info()

    # ----- UI -----
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Open Data…", command=self.on_open).pack(side=tk.LEFT)
        ttk.Button(top, text="Export CSV…", command=self.on_export).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(top, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=(6,0))
        self.file_label = ttk.Label(top, text="No file loaded"); self.file_label.pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(self.root); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: plot
        left = ttk.Frame(mid); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = plt.Figure(figsize=(7,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("Intensity")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.nav = NavigationToolbar2Tk(self.canvas, left)
        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click_plot)

        # Right: controls
        right = ttk.Frame(mid, padding=6); right.pack(side=tk.RIGHT, fill=tk.Y)

        # Baseline box
        baseline_box = ttk.Labelframe(right, text="Baseline (ALS)"); baseline_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(baseline_box, text="Apply baseline", variable=self.use_baseline, command=self.refresh_plot).pack(anchor="w")

        mode_row = ttk.Frame(baseline_box); mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Add to fit", variable=self.baseline_mode, value="add", command=self.refresh_plot).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Subtract",  variable=self.baseline_mode, value="subtract", command=self.refresh_plot).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(baseline_box, text="Baseline uses fit range", variable=self.baseline_use_range,
                        command=self.on_baseline_use_range_toggle).pack(anchor="w", pady=(2,0))

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
        self.tree = ttk.Treeview(peaks_box, columns=cols, show="headings", height=10, selectmode="browse")
        headers = ["#", "Center", "Height", "FWHM", "Lock W", "Lock C"]
        widths  = [30,   90,       90,       90,      70,       70]
        for c, txt, w in zip(cols, headers, widths):
            self.tree.heading(c, text=txt); self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)
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
        ttk.Checkbutton(inter, text="Add peaks on click", variable=self.add_peaks_mode).pack(anchor="w")
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
        solver_box = ttk.Labelframe(right, text="Solver"); solver_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(solver_box, textvariable=self.solver_var, state="readonly",
                     values=["Classic", "Modern", "LMFIT"], width=12).pack(side=tk.LEFT, padx=4)

        # Uncertainty panel
        unc_box = ttk.Labelframe(right, text="Uncertainty"); unc_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(unc_box, textvariable=self.unc_method, state="readonly",
                     values=["Asymptotic", "Bootstrap", "Bayesian"], width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(unc_box, text="Run", command=self.run_uncertainty).pack(side=tk.LEFT, padx=4)

        # Performance panel
        perf_box = ttk.Labelframe(right, text="Performance"); perf_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(perf_box, text="Numba", variable=self.perf_numba,
                        command=self.apply_performance).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="GPU", variable=self.perf_gpu,
                        command=self.apply_performance).pack(anchor="w")
        rowp = ttk.Frame(perf_box); rowp.pack(fill=tk.X, pady=2)
        ttk.Label(rowp, text="Seed:").pack(side=tk.LEFT)
        ttk.Entry(rowp, width=8, textvariable=self.seed_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(rowp, text="Max workers:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Spinbox(rowp, from_=0, to=64, textvariable=self.workers_var, width=5).pack(side=tk.LEFT)
        ttk.Button(rowp, text="Apply", command=self.apply_performance).pack(side=tk.LEFT, padx=4)

        # Actions
        actions = ttk.Labelframe(right, text="Actions"); actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Auto-seed", command=self.auto_seed).pack(side=tk.LEFT)
        ttk.Button(actions, text="Step \u25B6", command=self.step_once).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Fit", command=self.fit).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Toggle components", command=self.toggle_components).pack(side=tk.LEFT, padx=4)

        # Status
        self.status = ttk.Label(self.root, text="Open CSV/TXT/DAT, set baseline/range, add peaks, set η, then Fit.")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

    def _new_figure(self):
        self.ax.clear()
        self.ax.set_xlabel("x"); self.ax.set_ylabel("Intensity")
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
        self.file_label.config(text=Path(path).name)
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
        self.status.config(text="Loaded. Adjust baseline, (optionally) set fit range, add peaks; Fit.")

    # ----- Baseline -----
    def on_baseline_use_range_toggle(self):
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

        try:
            if mask is None or not np.any(mask):
                self.baseline = signals.als_baseline(self.y_raw, lam=lam, p=asym,
                                                     niter=niter, tol=thresh)
            else:
                x_sub = self.x[mask]
                y_sub = self.y_raw[mask]
                z_sub = signals.als_baseline(y_sub, lam=lam, p=asym,
                                             niter=niter, tol=thresh)
                self.baseline = np.interp(self.x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
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
        if self.span is not None:
            try:
                self.span.set_active(True)
                return
            except Exception:
                self.span = None

        def onselect(xmin, xmax):
            self.fit_xmin, self.fit_xmax = float(xmin), float(xmax)
            self.fit_min_var.set(f"{self.fit_xmin:.6g}")
            self.fit_max_var.set(f"{self.fit_xmax:.6g}")
            if self.span is not None:
                try:
                    self.span.set_active(False)
                except Exception:
                    pass
            if self.baseline_use_range.get():
                self.compute_baseline()
            else:
                self.refresh_plot()

        try:
            self.span = SpanSelector(self.ax, onselect, "horizontal", useblit=True,
                                     props=dict(alpha=0.15, facecolor="tab:blue"))
        except TypeError:
            self.span = SpanSelector(self.ax, onselect, "horizontal", useblit=True,
                                     rectprops=dict(alpha=0.15, facecolor="tab:blue"))
        self.status.config(text="Drag on the plot to select the fit x-range…")

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
        self.status.config(text=f"Applied template '{name}' with {len(self.peaks)} peak(s).")

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
        # Ignore clicks when zoom/pan active or toggle off
        nav = getattr(self, "nav", None)
        active = getattr(nav, "_active", None)
        mode = getattr(nav, "mode", "")
        if (active in ("PAN", "ZOOM")) or ("zoom" in str(mode).lower()) or ("pan" in str(mode).lower()):
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

        try:
            theta, _ = step_engine.step_once(
                x_fit, y_fit, self.peaks, mode, base_fit,
                loss="linear", weights=None, damping=0.0,
                trust_radius=np.inf, bounds=None
            )
        except Exception as e:
            messagebox.showerror("Step", f"Step failed:\n{e}")
            return

        j = 0
        for pk in self.peaks:
            c, h, w, eta = theta[j:j+4]; j += 4
            if not pk.lock_center:
                pk.center = float(c)
            pk.height = float(h)
            if not pk.lock_width:
                pk.fwhm = float(abs(w))
            pk.eta = float(eta)

        self.refresh_tree(keep_selection=True)
        self.refresh_plot()
        self.status.config(text="Step complete. Fit again or Export.")

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

        solver = self.solver_var.get().lower()
        options: dict = {}
        try:
            if solver == "modern":
                res = modern.solve(x_fit, y_fit, self.peaks, mode, base_fit, options)
            elif solver == "lmfit":
                res = lmfit_backend.solve(x_fit, y_fit, self.peaks, mode, base_fit, options)
            else:
                res = classic.solve(x_fit, y_fit, self.peaks, mode, base_fit, options)
        except Exception as e:
            messagebox.showerror("Fit", f"Fitting failed:\n{e}")
            return

        theta = res["theta"]
        for i, pk in enumerate(self.peaks):
            c, h, w, eta = theta[4*i:4*(i+1)]
            if not pk.lock_center:
                pk.center = float(c)
            pk.height = float(h)
            if not pk.lock_width:
                pk.fwhm = float(abs(w))
            pk.eta = float(eta)

        self.refresh_tree(keep_selection=True)
        self.refresh_plot()
        self.status.config(text="Fit complete. Edit/lock as needed; Fit again or Export.")

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
        method = self.unc_method.get().lower()
        try:
            if method == "asymptotic":
                rep = asymptotic.asymptotic({"theta": theta, "jac": None}, resid_fn)
            elif method == "bootstrap":
                cfg = {"x": x_fit, "y": y_fit, "peaks": self.peaks, "mode": mode,
                       "baseline": base_fit, "theta": theta, "options": {}, "n": 100}
                rep = bootstrap.bootstrap(self.solver_var.get().lower(), cfg, resid_fn)
            elif method == "bayesian":
                init = {"x": x_fit, "y": y_fit, "peaks": self.peaks, "mode": mode,
                        "baseline": base_fit, "theta": theta}
                rep = bayes.bayesian({}, "gaussian", init, {}, None)
            else:
                messagebox.showerror("Uncertainty", "Unknown method")
                return
        except Exception as e:
            messagebox.showerror("Uncertainty", f"Failed: {e}")
            return

        sigmas = rep.get("params", {}).get("sigma")
        if sigmas is not None:
            msg = "σ: " + ", ".join(f"{s:.3g}" for s in np.ravel(sigmas))
        else:
            msg = f"Computed {rep.get('type')} uncertainty."
        self.status.config(text=msg)
        messagebox.showinfo("Uncertainty", msg)

    def apply_performance(self):
        performance.set_numba(bool(self.perf_numba.get()))
        performance.set_gpu(bool(self.perf_gpu.get()))
        seed_txt = self.seed_var.get().strip()
        seed = int(seed_txt) if seed_txt else None
        performance.set_seed(seed)
        performance.set_max_workers(self.workers_var.get())
        self.status.config(text="Performance options applied.")

    def on_export(self):
        if self.x is None or self.y_raw is None or not self.peaks:
            messagebox.showinfo("Export", "Load data and perform a fit first.")
            return
        out_csv = filedialog.asksaveasfilename(
            title="Save peak table as CSV",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out_csv:
            return

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
        fname = self.file_label.cget("text")
        for i, (p, a) in enumerate(zip(self.peaks, areas), 1):
            rows.append({
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
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        # Trace CSV
        trace_path = str(Path(out_csv).with_name(Path(out_csv).stem + "_trace.csv"))
        df = pd.DataFrame({
            "x": self.x,
            "y_raw": self.y_raw,
            "baseline": base,
            "y_corr": y_corr,
            "y_target": y_target,   # data actually used for fitting
            "y_fit": y_fit
        })
        for k, v in comp_cols.items():
            df[k] = v
        df.to_csv(trace_path, index=False)

        messagebox.showinfo("Export", f"Saved:\n{out_csv}\n{trace_path}")

    # ----- Plot -----
    def toggle_components(self):
        self.components_visible = not self.components_visible
        self.refresh_plot()

    def refresh_plot(self):
        LW_RAW, LW_BASE, LW_CORR, LW_COMP, LW_FIT = 1.0, 1.0, 0.9, 0.8, 1.2
        self.ax.clear()
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
                for i, p in enumerate(self.peaks, 1):
                    comp = pseudo_voigt(self.x, p.height, p.center, p.fwhm, p.eta)
                    total_peaks += comp
                    comp_plot = (base + comp) if (base_applied and add_mode) else comp
                    self.ax.plot(self.x, comp_plot, lw=LW_COMP, alpha=0.6, label=f"Peak {i}")
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

        self.ax.set_xlabel("x"); self.ax.set_ylabel("Intensity")
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    # ----- Help -----
    def show_help(self):
        message = (
            "Supported files:\n"
            "  • CSV, TXT, DAT with two numeric columns (x,y).\n"
            "  • Delimiters auto-detected (comma, tab, spaces, semicolons).\n"
            "  • Lines starting with #, %, // (or text headers) are ignored.\n\n"
            "Baseline modes:\n"
            "  • Add to fit (default): model = baseline + sum(peaks) on raw data.\n"
            "  • Subtract: model = sum(peaks) on (raw - baseline).\n\n"
            "Fit range:\n"
            "  • Type Min/Max or drag with 'Select on plot'; 'Full range' clears.\n"
            "  • 'Baseline uses fit range' computes ALS on that slice, then interpolates.\n\n"
            "Templates:\n"
            "  • Save multiple named templates; 'Save changes' overwrites selected.\n"
            "  • 'Auto-apply on open' uses the currently selected template.\n\n"
            "Interaction:\n"
            "  • 'Add peaks on click' toggles picking; Zoom/Pan ignores clicks.\n"
            "  • 'Zoom out' widens x-view; 'Reset view' equals toolbar Home.\n"
        )
        messagebox.showinfo("Help", message)


def main():
    root = tk.Tk()
    app = PeakFitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
