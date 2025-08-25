# Interactive Peak Fit GUI (Gaussian–Lorentzian / pseudo-Voigt)

**Designed by** Farhan Zahin • **Built with** ChatGPT  
**Build:** v3.2-beta

> Cross-platform Tkinter + Matplotlib desktop app for interactive spectral peak fitting with ALS baseline, robust modern solvers (TRF / Variable-Projection), a simple Classic solver (curve_fit), single-step iteration (“Step ▶”), batch processing, persisted settings, and unified exports.

---

## Screenshot

> Replace with your image or keep the path below.

![Main window](docs/main-window.png)

---

## What’s new in v3.2-beta (vs. v3.0 / v3.1)

- **Reliability & parity**
  - Modern TRF/VP stabilized (robust losses/weights; VP solves heights via NNLS).
  - **Classic** restored as a simple **SciPy `curve_fit`** backend with lock-aware bounds.
- **“Step ▶” parity**  
  Uses the same residuals/bounds as Fit (modern & classic), with damping, backtracking, and NaN/Inf guards. Step accepts only when cost decreases.
- **Persisted UI settings**  
  Shape factor (η), “Add peaks on click”, solver choice, and uncertainty method survive restarts.
- **UI/UX polish**  
  Right panel scroll isolated from Help window; resizable splitter with sensible min widths; bottom status bar with progress and live log; batch streams progress there.
- **Uncertainty (asymptotic)**  
  Quick CI band computed from the Jacobian/covariance at the solution (bootstrap/MCMC planned).

> The previous **v2.7** stable standalone release remains available.

---

## Features

- **Data loading**
  - Robust import for **CSV/TXT/DAT** (2 columns x,y). Delimiters auto-detected; lines with `# % //` and text headers ignored. Data sorted if x is descending.
- **ALS baseline (Eilers–Boelens)**
  - Parameters: **λ (smooth)**, **p (asym)**, **Iterations**, **Threshold** (optional early stop).
  - Optionally compute ALS **within the fit window only** then interpolate to full x.
- **Fit modes**
  - **Add to fit**: model = baseline + Σ(peaks) vs raw y (WYSIWYG plotting).
  - **Subtract**: model = Σ(peaks) vs (y − baseline).
- **Interactive peaks**
  - Click to add peaks (toggle), or **Auto-seed** prominent peaks in the active window.
  - Per-peak **lock Width** / **lock Center**; per-peak **η** (Gaussian–Lorentzian mix) or **Apply to all**.
- **Solvers**
  - **Classic (curve_fit)**: simple unweighted least squares with minimal bounds; honors locks.
  - **Modern TRF**: SciPy Trust-Region-Reflective with bounds; **loss** (`linear`, `soft_l1`, `huber`, `cauchy`) and **weights** (`none`, `poisson`, `inv_y`).
  - **Modern VP**: Variable-Projection — heights solved fast via NNLS; same robust loss/weights.
  - **LMFIT-VP (optional)**: if `lmfit` is installed.
  - **Step ▶**: one damped iteration that only commits on cost decrease (shared residuals/bounds).
- **Batch processing**
  - Folder patterns (`*.csv;*.txt;*.dat`), seed from current / template / auto; optional per-file **re-height**; one summary CSV; optional per-spectrum **trace CSVs**.
- **Persisted configuration**
  - Stored in `~/.gl_peakfit_config.json`: baseline defaults, batch defaults, templates, auto-apply, x-label, and key UI settings.
- **Exports (unified schemas)**
  - Peak Table CSV and Trace CSV (both single run and batch).

---

## Install

### Option A — pip (virtualenv recommended)
```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib lmfit

Option B — conda
conda install numpy scipy pandas matplotlib
pip install lmfit


lmfit is optional and only needed for the LMFIT backend.

Run
python run_app.py
```

The main window opens with the plot on the left and controls on the right.

Help dialog (F1) is scrollable and independent of the right-panel scroll.

Quick Start

Open Data… and select a two-column file (x,y).

Adjust Baseline (ALS) parameters (λ, p, Iterations, Threshold).
Optionally check “Baseline uses fit range” and set a range (drag or enter Min/Max).

Add peaks by clicking near their centers (or use Auto-seed).

(Optional) Lock Width or Center, set per-peak η, or Apply to all.

Choose Fit mode (Add/Subtract) and Fitting method (Classic / Modern TRF / Modern VP / LMFIT-VP).

Click Fit. Use Step ▶ to inspect single-iteration behavior.

Export CSV (peak table and trace CSV) or run Batch for a folder.

Uncertainty

Asymptotic: computes covariance from the Jacobian at the solution and overlays a 95% CI band on the fitted curve.

Bootstrap / MCMC: planned for future releases.

Export Schemas
A) Peak Table CSV (single & batch use the same columns and order)
```
file, peak, center, height, fwhm, eta, lock_width, lock_center,
area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax
```

area: pseudo-Voigt closed-form area

area_pct: area normalized by total area

rmse: computed over the active fit window versus the correct target (Add: raw y; Subtract: y − baseline)

B) Trace CSV (per spectrum; single & batch)
```
x, y_raw, baseline,
y_target_add, y_fit_add, peak1, peak2, …,
y_target_sub, y_fit_sub, peak1_sub, peak2_sub, …
```

peakN = baseline-ADDED component (matches Add-mode display)

peakN_sub = baseline-SUBTRACTED pure component (useful for calculations)

Batch Processing

Select folder & patterns (semicolon-separated).

Peaks source: Current (optionally re-height), Selected template, or Auto-seed.

Writes one summary CSV and, if enabled, per-spectrum trace CSVs.

Progress and per-file messages stream to the status bar log.

Tips & Troubleshooting

Add vs. Subtract: In Add mode, the baseline is added back during fitting. If fits look too tall, double-check the mode.

ALS tuning: Increase λ (smoother) and/or decrease p (more under peaks) if ALS rides peaks.

Robustness: For spikes/outliers, try robust loss (soft_l1, huber, cauchy) and/or weights (poisson or inv_y).

Step ▶ rejected: Try a smaller damping λ or run a full Fit to re-linearize.

Scroll: The right panel scrolls when the mouse is over it; the Help window has its own scroll.

Packaging (optional)

Create a single-file executable with PyInstaller:

pip install pyinstaller
pyinstaller -F -n PeakFit --add-data "ui;ui" --add-data "docs;docs" run_app.py
# On macOS you may prefer: pyinstaller -w -F ...
# On Windows, ensure tcl/tk resources are found; PyInstaller usually bundles them automatically.


The output binary will be in dist/PeakFit (or PeakFit.exe on Windows).

Version History (high level)

v3.2-beta – Stability and parity; Classic restored (curve_fit); Step ▶ unified & safe; settings persisted; status/log UI; asymptotic uncertainty band.

v3.1 – Solver tuning: center-in-window (optional), Δx-based FWHM lower bound, parameter-wise jitter, x-scaling.

v3.0 – Modern TRF with robust losses & weights; ALS iterations/threshold; scrollable Help; resizable panel; persisted x-label.

v2.7 – Stable standalone release with ALS/Help/x-label persistence.

… earlier versions: template system, range selection, exports, etc.

License

MIT — see LICENSE.

Citation

If this tool helps your research, please consider citing this repository:

Zahin, F. (2025). Interactive Peak Fit GUI (pseudo-Voigt). GitHub repository.
