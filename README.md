# Interactive-Peak-Fit-GUI
Interactive peak fit GUI for any spectroscopy (Gaussian–Lorentzian / pseudo-Voigt)

Designed by Farhan Zahin, 
built with ChatGPT

Build: v3 (axes label editing with superscript/subscript, scrollable Help, ALS iterations/threshold,
solver backends with robust losses/weights, Step iteration, uncertainty/performance panels)

The previous v2.7 stable release remains available for download and can run standalone.

<img width="1987" height="1289" alt="main-window png" src="https://github.com/user-attachments/assets/9c0b832c-3355-49e7-9bee-01be2abbad60" />

Overview
========
An interactive desktop GUI (Tkinter + Matplotlib) that lets you:

• Load 2-column spectra from CSV/TXT/DAT (x, y). Delimiters auto-detected; lines with #, %, // and text headers are ignored.

• Estimate a baseline with ALS (λ smoothing, p asymmetry), now with:
    - Iterations: stop after N passes (stability).
    - Threshold: optional early-stop if the baseline changes less than a tolerance (speed).

• Fit either:
    - Add to fit  : model = baseline + Σ(peaks) against raw data (WYSIWYG plotting).
    - Subtract    : model = Σ(peaks) against (raw − baseline).

• Optionally compute the ALS baseline **only inside** the selected fit range, then interpolate across the full x.

• Add peaks by clicking (toggle), or auto-seed prominent peaks in the current range.

• Lock width and/or center per peak; fit solves height + any UNLOCKED params (η is user-set per peak or broadcasted globally).

• Choose Gaussian–Lorentzian mix (η, 0..1). “Apply to all” copies η to every peak.

• Choose/clear a fit x-range by typing limits or dragging on the plot. Shaded region indicates the active window.

• Thin line rendering and “Toggle components” for clarity during inspection.

• Select a fitting engine:
    - **Classic** – fast linear least squares that only adjusts peak heights; ideal when the baseline is trustworthy and peaks are well seeded.
    - **Modern** – SciPy’s Trust Region Reflective solver for full non-linear fitting with support for robust loss functions (`linear`, `soft_l1`, `huber`), optional per-point weights (`1/σ²`), configurable multi-start restarts and random jitter to escape local minima.
    - **Step ▶** – executes a single Gauss–Newton iteration so you can visualize convergence step-by-step.

• Uncertainty & performance tools:
    - **Uncertainty**
        * *Asymptotic* – covariance from the solver’s final Jacobian.
        * *Bootstrap* – residual resampling with configurable iterations, parallel workers and reproducible seeds.
        * *Bayesian* – MCMC sampling via `emcee` to obtain posterior distributions and credible intervals.
    - **Performance**
        * Toggle Numba JIT or CuPy GPU acceleration for faster model evaluations.
        * Cache ALS baselines between fits to skip repeated computations.
        * Set deterministic seeds for reproducible restarts and bootstrap draws.
        * Run bootstrap iterations in parallel to utilize multiple CPU cores.

• Axes/Labels: set a custom X-axis label with superscript/subscript helpers and save it as default (persists in ~/.gl_peakfit_config.json).

• Peak templates: save as new, save changes, apply, delete; optional auto-apply on file open.

• **Batch processing over folders with patterns (*.csv;*.txt;*.dat). Seed from current/template/auto. Optional re-height per file. Optional per-spectrum trace exports. One summary CSV.**

• Scrollable right-side control panel (mouse-wheel works anywhere on the panel).

• Configuration persisted in ~/.gl_peakfit_config.json (baseline defaults, batch defaults, templates, auto-apply, x-label).

Data Exports
============
A) Peak table CSV (single export AND batch summary; identical columns/order):

  file, peak, center, height, fwhm, eta, lock_width, lock_center,

  area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax

B) Trace CSV (per spectrum; identical schema for single and batch *_trace.csv):

  x, y_raw, baseline,

  y_target_add, y_fit_add, peak1, peak2, …,

  y_target_sub, y_fit_sub, peak1_sub, peak2_sub, …

Where:

  • peakN     = baseline-ADDED component (for plotting like “Add” mode)

  • peakN_sub = baseline-SUBTRACTED pure component (for calculations)

Keyboard/Mouse Tips
===================

• Toolbar Zoom/Pan disables click-to-add-peaks automatically (so zooming never creates peaks).

• “Add peaks on click” is a toggle; turn it off when editing.

• Use “Select on plot” to drag a fit window; “Full range” clears it.

• “Zoom out” and “Reset view” give quick navigation.

Version History (high-level)
============================

v3    – Superscript/subscript helpers for axis labels; scrollable Help dialog; ALS baseline exposes Iterations and Threshold; mouse-wheel scrolling on the right panel; solver backends (Classic/Modern), Step iteration, uncertainty and performance panels.

v2.7  – Previous stable standalone release (still available for download).

v2.6  – Unified single/batch **peak table** schema and metadata via shared builder; identical column order everywhere.

v2.5  – Trace CSV contains **both sections** (added & subtracted) in fixed order: y_target_add/y_fit_add/peakN + y_target_sub/y_fit_sub/peakN_sub.

v2.4  – Single export & batch use a unified **trace builder** so formats can’t drift.

v2.3  – Fixed potential height inflation in Add mode during batch by always adding the baseline slice in the optimizer.

v2.2  – Exported both baseline-added and baseline-subtracted components.

v2.1  – Scrollable right-side control panel; better small-screen usability.

v2.0  – Batch/mapping: folder patterns, template/current/auto seeding, re-height option, per-spectrum traces, summary CSV.

v1.9  – Click-add toggle decoupled from zoom so zooming never adds peaks by mistake.

v1.8  – “Add to fit” mode to fit **over** the baseline (WYSIWYG) in addition to subtract mode.

v1.7  – Zoom-out and reset-view buttons.

v1.6  – Multiple peak templates + “save changes” to an existing template.

v1.5  – Lock **center** as well as width.

v1.4  – Baseline can be computed using only the selected fit range.

v1.3  – Fit-range selection on the plot (SpanSelector) and via numeric fields.

v1.2  – Components plotted on top of baseline in Add mode; thinner line styles for readability.

v1.1  – Lock width; per-peak η; apply η to all peaks.

v1.0  – First stable GUI with ALS baseline, GL peaks, iterative fitting, CSV export.
