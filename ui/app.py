#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive peak fit GUI for spectra (Gaussian–Lorentzian / pseudo-Voigt)
Designed by Farhan Zahin
Built with ChatGPT

Build: v3

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
  • Solver selection (Classic (curve_fit), Modern, LMFIT) plus Step ▶ single iteration
  • Multiple peak templates (save as new, save changes, select/apply, delete); optional auto-apply on open
  • Zoom out & Reset view buttons
  • Supports CSV, TXT, DAT (auto delimiter detection; skips headers/comments)
  • Export peak table with metadata and full trace CSV (raw, baseline, components)
"""

import json
import math
import re
import time
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
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import threading
import traceback

from scipy.signal import find_peaks

from core import signals
from core.residuals import build_residual, jacobian_fd
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
    "ui_theme": "Light",
}

LOG_MAX_LINES = 5000

PALETTE = {
    "Light": {
        "bg": "#ffffff",
        "panel": "#f3f3f3",
        "fg": "#000000",
        "accent": "#268bd2",
        "grid": "#e0e0e0",
        "line": "#000000",
        "tick": "#000000",
    },
    "Dark": {
        "bg": "#1e1e1e",
        "panel": "#2a2a2a",
        "fg": "#e6e6e6",
        "accent": "#4aa3ff",
        "grid": "#3a3a3a",
        "line": "#e6e6e6",
        "tick": "#e6e6e6",
    },
}


def _toolbar_restyle(toolbar, pal):
    try:
        toolbar.configure(background=pal["panel"])
    except Exception:
        pass
    for child in toolbar.winfo_children():
        try:
            child.configure(
                background=pal["panel"],
                activebackground=pal["bg"],
                foreground=pal["fg"],
                activeforeground=pal["fg"],
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
                padx=3,
                pady=2,
            )
        except Exception:
            pass


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
            # Migration: theme -> ui_theme
            if "ui_theme" not in cfg and "theme" in cfg:
                cfg["ui_theme"] = str(cfg["theme"]).title()
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

        tmp = re.sub(r"(?<!\$)\^\s*\{([^{}]+)\}", lambda m: "$^{" + m.group(1) + "}$", tmp)
        tmp = re.sub(r"(?<!\$)\^\s*([+\-]?\d+(?:\.\d+)?)", lambda m: "$^{" + m.group(1) + "}$", tmp)
        tmp = re.sub(r"(?<!\$)\^\s*(\w+)", lambda m: "$^{" + m.group(1) + "}$", tmp)
        tmp = re.sub(r"(?<!\$)_\s*\{([^{}]+)\}", lambda m: "$_{" + m.group(1) + "}$", tmp)
        tmp = re.sub(r"(?<!\$)_(\w+)", lambda m: "$_{" + m.group(1) + "}$", tmp)

        tmp = tmp.replace(ESC_CARET, "^").replace(ESC_UND, "_")
        out.append(tmp)

    return "".join(out)

# ---------- Scrollable frame ----------
class ScrollableFrame(ttk.Frame):
    """A ttk.Frame that contains a vertically scrollable interior frame."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.canvas.configure(background="#FFFFFF")
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self.canvas)
        self.interior.configure(background="#FFFFFF")
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


# ---------- Main GUI ----------
class PeakFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Origin-like Peak Fit (pseudo-Voigt)")

        self._native_theme = ttk.Style().theme_use()

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
        self.global_eta = tk.DoubleVar(value=self.cfg.get("ui_eta", 0.5))
        self.global_eta.trace_add("write", lambda *_: self._on_eta_change())
        self.auto_apply_template = tk.BooleanVar(value=bool(self.cfg.get("auto_apply_template", False)))
        self.auto_apply_template_name = tk.StringVar(value=self.cfg.get("auto_apply_template_name", ""))

        # Interaction
        self.add_peaks_mode = tk.BooleanVar(value=bool(self.cfg.get("ui_add_peaks_on_click", True)))
        self.ui_theme = tk.StringVar(value=self.cfg.get("ui_theme", "Light"))

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
        self.components_visible = True

        # Matplotlib click binding
        self.cid = None

        # Axis label
        self.x_label_var = tk.StringVar(value=str(self.cfg.get("x_label", "x")))
        self.x_label_auto_math = tk.BooleanVar(value=bool(self.cfg.get("x_label_auto_math", True)))

        # Batch defaults
        self.batch_patterns = tk.StringVar(value=self.cfg.get("batch_patterns", "*.csv;*.txt;*.dat"))
        self.batch_source = tk.StringVar(value=self.cfg.get("batch_source", "template"))
        self.batch_reheight = tk.BooleanVar(value=bool(self.cfg.get("batch_reheight", False)))
        self.batch_auto_max = tk.IntVar(value=int(self.cfg.get("batch_auto_max", 5)))
        self.batch_save_traces = tk.BooleanVar(value=bool(self.cfg.get("batch_save_traces", False)))

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

        self.show_ci_band = False
        self.ci_band = None
        self.show_ci_band_var = tk.BooleanVar(value=False)

        # Uncertainty and performance controls
        unc_cfg = self.cfg.get("unc_method", "asymptotic")
        if unc_cfg == "bootstrap":
            unc_label = f"Bootstrap (base solver = {SOLVER_LABELS[self.solver_choice.get()]})"
        elif unc_cfg == "bayesian":
            unc_label = "Bayesian"
        else:
            unc_label = "Asymptotic"
        self.unc_method = tk.StringVar(value=unc_label)
        self.perf_numba = tk.BooleanVar(value=False)
        self.perf_gpu = tk.BooleanVar(value=False)
        self.perf_cache = tk.BooleanVar(value=True)
        self.perf_deterministic = tk.BooleanVar(value=False)
        self.perf_parallel = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="")
        self.workers_var = tk.IntVar(value=0)
        self.gpu_chunk_var = tk.IntVar(value=262144)

        # UI
        self._build_ui()
        self.apply_theme(self.ui_theme.get())
        self._new_figure()
        self._update_template_info()

    # ----- UI -----
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Open Data…", command=self.on_open).pack(side=tk.LEFT)
        ttk.Button(top, text="Export CSV…", command=self.on_export).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(top, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=(6,0))
        self.file_label = ttk.Label(top, text="No file loaded"); self.file_label.pack(side=tk.LEFT, padx=10)

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
        right_scroll.pack(fill=tk.BOTH, expand=True)
        self.right_scroll = right_scroll
        right = right_scroll.interior
        right.configure(padding=6)

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

        # Axes / label controls
        axes_box = ttk.Labelframe(right, text="Axes / Labels")
        axes_box.pack(fill=tk.X, pady=6)
        ttk.Label(axes_box, text="X-axis label:").pack(side=tk.LEFT)
        self.x_label_entry = ttk.Entry(axes_box, width=16, textvariable=self.x_label_var)
        self.x_label_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(axes_box, text="Apply", command=self.apply_x_label).pack(side=tk.LEFT, padx=2)
        ttk.Button(axes_box, text="Superscript", command=self.insert_superscript).pack(side=tk.LEFT, padx=2)
        ttk.Button(axes_box, text="Subscript", command=self.insert_subscript).pack(side=tk.LEFT, padx=2)
        ttk.Button(axes_box, text="Save as default", command=self.save_x_label_default).pack(side=tk.LEFT, padx=2)

        row_fmt = ttk.Frame(axes_box)
        row_fmt.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(row_fmt, text="Auto-format superscripts/subscripts", variable=self.x_label_auto_math,
                        command=self._on_x_label_auto_math_toggle).pack(anchor="w")

        row_theme = ttk.Frame(axes_box)
        row_theme.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row_theme, text="Theme:").pack(side=tk.LEFT)
        self.theme_combo = ttk.Combobox(
            row_theme,
            textvariable=self.ui_theme,
            state="readonly",
            width=8,
            values=("Light", "Dark"),
        )
        self.theme_combo.pack(side=tk.LEFT, padx=4)
        self.theme_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self.apply_theme(self.ui_theme.get())
        )

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
        ttk.Button(unc_box, text="Run", command=self.run_uncertainty).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            unc_box,
            text="Show uncertainty band",
            variable=self.show_ci_band_var,
            command=self._toggle_ci_band,
        ).pack(anchor="w", padx=4)
        self._update_unc_widgets()

        # Performance panel
        perf_box = ttk.Labelframe(right, text="Performance"); perf_box.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(perf_box, text="Numba", variable=self.perf_numba,
                        command=self.apply_performance).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="GPU", variable=self.perf_gpu,
                        command=self.apply_performance).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Cache baseline", variable=self.perf_cache,
                        command=self.apply_performance).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Deterministic seeds", variable=self.perf_deterministic,
                        command=self.apply_performance).pack(anchor="w")
        ttk.Checkbutton(perf_box, text="Parallel bootstrap", variable=self.perf_parallel,
                        command=self.apply_performance).pack(anchor="w")
        rowp = ttk.Frame(perf_box); rowp.pack(fill=tk.X, pady=2)
        ttk.Label(rowp, text="Seed:").pack(side=tk.LEFT)
        ttk.Entry(rowp, width=8, textvariable=self.seed_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(rowp, text="Max workers:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Spinbox(rowp, from_=0, to=64, textvariable=self.workers_var, width=5).pack(side=tk.LEFT)
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
        ttk.Button(batch_box, text="Run Batch…", command=self.run_batch).pack(side=tk.LEFT, pady=4)

        # Actions
        actions = ttk.Labelframe(right, text="Actions"); actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Auto-seed", command=self.auto_seed).pack(side=tk.LEFT)
        self.step_btn = ttk.Button(actions, text="Step \u25B6", command=self.step_once)
        self.step_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Fit", command=self.fit).pack(side=tk.LEFT, padx=4)
        ttk.Label(actions, textvariable=self.solver_title).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Toggle components", command=self.toggle_components).pack(side=tk.LEFT, padx=4)

        # Status bar and log
        bar = ttk.Frame(self.root); bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Open CSV/TXT/DAT, set baseline/range, add peaks, set η, then Fit.")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)
        self.log_btn = ttk.Button(bar, text="Show log \u25B8", command=self.toggle_log)
        self.log_btn.pack(side=tk.RIGHT)
        self.pbar = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pbar.pack(side=tk.RIGHT, padx=6)
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
        else:
            self.bootstrap_solver_combo.pack_forget()

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

    def _restyle_toolbar(self, pal):
        if hasattr(self, "nav"):
            _toolbar_restyle(self.nav, pal)

    def apply_theme(self, mode: str):
        pal = PALETTE.get(mode, PALETTE["Light"])
        style = ttk.Style()
        if mode == "Light":
            style.theme_use(self._native_theme)
            for sty in (
                "TFrame",
                "TLabelframe",
                "TLabelframe.Label",
                "TLabel",
                "TButton",
                "TCheckbutton",
                "TRadiobutton",
                "TEntry",
                "TCombobox",
                "Vertical.TScrollbar",
                "Horizontal.TScrollbar",
                "Treeview",
                "Treeview.Heading",
                "TProgressbar",
            ):
                style.configure(sty, background="", foreground="", fieldbackground="", insertcolor="", troughcolor="")
                style.map(sty, background=[], foreground=[])
            self.root.option_add("*TCombobox*Listbox*Background", "")
            self.root.option_add("*TCombobox*Listbox*Foreground", "")
        else:
            style.theme_use("clam")
            style.configure("TFrame", background=pal["panel"])
            style.configure("TLabelframe", background=pal["panel"], foreground=pal["fg"])
            style.configure("TLabelframe.Label", background=pal["panel"], foreground=pal["fg"])
            style.configure("TLabel", background=pal["panel"], foreground=pal["fg"])
            style.configure("TButton", background=pal["panel"], foreground=pal["fg"], padding=(6, 3))
            style.map("TButton", background=[("active", pal["bg"])])
            style.configure("TCheckbutton", background=pal["panel"], foreground=pal["fg"])
            style.configure("TRadiobutton", background=pal["panel"], foreground=pal["fg"])
            style.configure("TEntry", fieldbackground=pal["bg"], foreground=pal["fg"])
            style.configure(
                "TCombobox",
                fieldbackground=pal["bg"],
                foreground=pal["fg"],
                background=pal["panel"],
            )
            style.configure("Vertical.TScrollbar", background=pal["panel"])
            style.configure("Horizontal.TScrollbar", background=pal["panel"])
            style.configure(
                "Treeview",
                background=pal["bg"],
                fieldbackground=pal["bg"],
                foreground=pal["fg"],
            )
            style.configure(
                "Treeview.Heading",
                background=pal["panel"],
                foreground=pal["fg"],
            )
            style.configure("TProgressbar", background=pal["accent"], troughcolor=pal["panel"])
            self.root.option_add("*TCombobox*Listbox*Background", pal["bg"])
            self.root.option_add("*TCombobox*Listbox*Foreground", pal["fg"])

        try:
            self.right_scroll.canvas.configure(background=pal["panel"])
            self.right_scroll.interior.configure(style="TFrame")
        except Exception:
            pass
        try:
            self.root.configure(background=pal["bg"])
        except Exception:
            pass
        if getattr(self, "_log_console", None) is not None:
            try:
                self._log_console.configure(background=pal["panel"], foreground=pal["fg"])
            except Exception:
                pass
        self.fig.patch.set_facecolor(pal["bg"])
        self.ax.set_facecolor(pal["bg"])
        self.ax.tick_params(colors=pal["tick"])
        for spine in self.ax.spines.values():
            spine.set_color(pal["fg"])
        self.ax.xaxis.label.set_color(pal["fg"])
        self.ax.yaxis.label.set_color(pal["fg"])
        self.ax.title.set_color(pal["fg"])
        self.ax.grid(color=pal["grid"])
        self._restyle_toolbar(pal)
        self.canvas.draw_idle()
        self.cfg["ui_theme"] = mode
        save_config(self.cfg)

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

    def _toggle_ci_band(self):
        self.show_ci_band = bool(self.show_ci_band_var.get())
        self.refresh_plot()

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
        self.status_var.set("Loaded. Adjust baseline, (optionally) set fit range, add peaks; Fit.")

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
        self.x_label_entry.insert(tk.INSERT, "^{ }")
        self.x_label_entry.icursor(self.x_label_entry.index(tk.INSERT) - 2)
        self.x_label_entry.focus_set()

    def insert_subscript(self):
        self.x_label_entry.insert(tk.INSERT, "_{ }")
        self.x_label_entry.icursor(self.x_label_entry.index(tk.INSERT) - 2)
        self.x_label_entry.focus_set()

    def apply_x_label(self):
        label = format_axis_label_inline(self.x_label_var.get(), self.x_label_auto_math.get())
        self.ax.set_xlabel(label)
        self.canvas.draw_idle()

    def save_x_label_default(self):
        self.cfg["x_label"] = self.x_label_var.get()
        save_config(self.cfg)
        messagebox.showinfo("Axes", f'Saved default x-axis label: "{self.x_label_var.get()}"')

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

        backend = BACKENDS.get(solver)
        if backend is None:  # pragma: no cover
            messagebox.showerror("Step", f"Unknown solver {solver}")
            return

        self.set_busy(True, "Stepping…")
        try:
            prep = backend.prepare_state(x_fit, y_fit, self.peaks, mode, base_fit, options)
            state = prep["state"]
            state, accepted, c0, c1, info = backend.iterate(state)
        except Exception as e:  # pragma: no cover - UI feedback only
            self.set_busy(False, "Step failed.")
            messagebox.showerror("Step", f"Step failed:\n{e}")
            self.log(f"Step failed: {e}", level="ERROR")
            return

        if accepted:
            theta = state.get("theta")
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
            msg = (
                f"{solver} Step accepted: Δcost={c0 - c1:.3g}, "
                f"backtracks={info.get('backtracks',0)}, λ={info.get('lambda',0.0):.3g}"
            )
        else:
            msg = f"{solver} Step rejected (reason={info.get('reason','no_decrease')})"
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
            if self.show_ci_band:
                self._run_asymptotic_uncertainty()
            else:
                self.ci_band = None
                self.show_ci_band = False
                self.show_ci_band_var.set(False)
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
            "peak_output": out_csv,
            "source": source,
            "reheight": bool(self.batch_reheight.get()),
            "auto_max": int(self.batch_auto_max.get()),
            solver: self._solver_options(solver),
        }

        self.cfg["batch_patterns"] = self.batch_patterns.get()
        self.cfg["batch_source"] = source
        self.cfg["batch_reheight"] = bool(self.batch_reheight.get())
        self.cfg["batch_auto_max"] = int(self.batch_auto_max.get())
        self.cfg["batch_save_traces"] = bool(self.batch_save_traces.get())
        save_config(self.cfg)

        def work():
            def prog(i, total, path):
                self.root.after(0, lambda: self.status_var.set(f"Batch {i}/{total}: {Path(path).name}"))

            return batch_runner.run(patterns, cfg, progress=prog, log=self.log_threadsafe)

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
            messagebox.showinfo("Batch", f"Summary saved:\n{out_csv}")

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
        self.ci_band = (x_all, lo, hi)

        bw = (hi - lo)[mask]
        bw_stats = (float(np.min(bw)), float(np.median(bw)), float(np.max(bw)))

        info = {
            "m": m,
            "n": theta.size,
            "rank": rank,
            "dof": m - rank,
            "cond": cond,
            "rmse": math.sqrt(rss / m),
            "bw": bw_stats,
            "warn_nonfinite": warn_nonfinite,
        }
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
                return self._run_asymptotic_uncertainty()
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
                }
                return bootstrap.bootstrap(self.bootstrap_solver_choice.get(), cfg, resid_fn)
            if method == "bayesian":
                init = {"x": x_fit, "y": y_fit, "peaks": self.peaks, "mode": mode,
                        "baseline": base_fit, "theta": theta}
                return bayes.bayesian({}, "gaussian", init, {}, resid_fn)
            raise RuntimeError("Unknown method")

        def done(res, err):
            if err or res is None:
                self.set_busy(False, "Uncertainty failed.")
                if err:
                    self.log(f"Uncertainty failed: {err}", level="ERROR")
                    messagebox.showerror("Uncertainty", f"Failed: {err}")
                return
            if method == "asymptotic":
                cov, theta, info = res
                self.show_ci_band = True
                self.show_ci_band_var.set(True)
                self.refresh_plot()
                lines, warns = self._format_asymptotic_summary(cov, theta, info, self.ci_band)
                for ln in lines:
                    self.log(ln)
                for ln in warns:
                    self.log(ln, level="WARN")
            else:
                sigmas = res.get("params", {}).get("sigma") if isinstance(res, dict) else None
                if sigmas is not None:
                    msg = "σ: " + ", ".join(f"{s:.3g}" for s in np.ravel(sigmas))
                else:
                    msg = f"Computed {getattr(res, 'type', 'unknown')} uncertainty."
                self.log(msg)
                messagebox.showinfo("Uncertainty", msg)
            self.set_busy(False, "Uncertainty ready (95% band).")

        self.set_busy(True, "Computing uncertainty…")
        self.run_in_thread(work, done)

    def apply_performance(self):
        performance.set_numba(bool(self.perf_numba.get()))
        performance.set_gpu(bool(self.perf_gpu.get()))
        performance.set_cache_baseline(bool(self.perf_cache.get()))
        seed_txt = self.seed_var.get().strip()
        seed = int(seed_txt) if seed_txt else None
        if self.perf_deterministic.get():
            performance.set_seed(seed)
        else:
            performance.set_seed(None)
        if self.perf_parallel.get():
            performance.set_max_workers(self.workers_var.get())
        else:
            performance.set_max_workers(0)
        performance.set_gpu_chunk(self.gpu_chunk_var.get())
        self.status_var.set("Performance options applied.")

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

        if self.show_ci_band and self.ci_band is not None:
            xb, lob, hib = self.ci_band
            self.ax.fill_between(xb, lob, hib, alpha=0.18, label="Uncertainty band")

        self.ax.legend(loc="best")
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
