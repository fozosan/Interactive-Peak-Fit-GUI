# Interactive Peak Fit GUI (Gaussian–Lorentzian / pseudo-Voigt)

**Designed by** Farhan Zahin • **Built with** ChatGPT  
**Build:** v3.2-beta

> Cross-platform Tkinter + Matplotlib desktop app for interactive spectral peak fitting with ALS baseline; Classic, Modern TRF, and Variable-Projection solvers; a single-step (“Step ▶”) debugger; **asymptotic/Bootstrap/Bayesian** uncertainty; batch processing; persisted settings; and unified exports (`*_fit.csv`, `*_trace.csv`, `*_uncertainty.csv/.txt`).

---

## Screenshot

<img width="1513" height="1141" alt="image" src="https://github.com/user-attachments/assets/60b49da0-401c-41b4-8d11-827650619279" />

---

## What’s new in v3.2-beta (vs. v3.0 / v3.1)

- **Reliability & parity**
  - Modern TRF/VP stabilized (robust losses/weights; VP heights via NNLS).
  - **Classic** restored as a simple **SciPy `curve_fit`** backend with lock-aware bounds.
  - **Single vs Batch parity**: unified fit engine; locks/bounds honored; baseline computed inside per-file window; deterministic seeding.
- **“Step ▶” parity**
  - Uses the **same residuals/weights/bounds** as Fit (Classic/TRF/VP).
  - Reports damping λ, backtracks, step_norm, and accept/reject reason with NaN/Inf guards.
- **Uncertainty**
  - **Asymptotic** 95% CI band (default) from Jacobian/covariance + delta method (smoothed for display).
  - **Bootstrap** residual resampling (seeded, lock/bounds aware) → param stats + optional prediction band.
  - **Bayesian** (optional `emcee`) posterior mean/SD and 95% credible intervals; ESS/R-hat diagnostics; predictive band. If `emcee` is missing, the app reports **NotAvailable**.
  - **Exports** now include `*_uncertainty.csv` (tabular) and `*_uncertainty.txt` (human-readable with **±** values).
- **UI/UX**
  - **Action bar** (top-right) segmented: **File** [Open, Export, Batch] | **Fit** [Step, Fit] | **Plot** [Uncertainty, Legend, Components] | **Help** (F1).
  - Right panel scroll isolated from Help; resizable splitter with sensible mins; status bar with progress + green-on-black log; Arial legend/font.
- **Persistence**
  - Saves to `~/.gl_peakfit_config.json`: baseline defaults, solver choice & options, uncertainty method, “Add peaks on click”, global η, legend toggle, batch defaults, x-label, **performance toggles** (Numba/GPU/cache/seed_all/max_workers), and templates (including **auto-apply**).

> The previous **v2.7** stable standalone release remains available.

---

## Features

- **Data loading**  
  Robust import for **CSV/TXT/DAT** (2 columns x,y). Delimiters auto-detected; lines with `# % //` and text headers ignored. Non-numeric columns dropped; non-finite rows removed; x sorted ascending if needed.

 - **Baseline correction (ALS or Polynomial)**
  ALS (λ, p, iterations, threshold) or polynomial (degree, optional normalization). Optionally compute the baseline **within the fit window** then interpolate to full x.

- **Fit modes**
  - **Add**: model = baseline + Σ(peaks) vs raw y (WYSIWYG plotting).
  - **Subtract**: model = Σ(peaks) vs (y − baseline).

- **Interactive peaks**  
  Click to add (toggle), or **Auto-seed** in the window. Per-peak **lock Width/Center**; per-peak η (Gaussian–Lorentzian mix) with **Apply to all**.

- **Solvers**
  - **Classic (curve_fit)**: simple unweighted least squares with minimal bounds; honors locks.
  - **Modern TRF**: SciPy Trust-Region-Reflective with bounds; **loss** (`linear`, `soft_l1`, `huber`, `cauchy`) and **weights** (`none`, `poisson`, `inv_y`).
  - **Modern VP**: Variable-Projection — heights solved by **NNLS**, centers/widths updated by damped Gauss–Newton with backtracking; same robust options.
  - **LMFIT-VP (optional)**: if `lmfit` is installed.
  - **Step ▶**: one damped iteration that only commits on cost decrease (shared residuals/weights/bounds with Fit).

- **Uncertainty**
  - **Asymptotic, Bootstrap, Bayesian** estimators; 95% bands/intervals; human-readable report with **±**; tabular CSV for downstream analysis.

- **Batch processing**
  - Folder patterns (`*.csv;*.txt;*.dat`), seed from current/template (**auto-apply**) / auto; optional per-file **re-height**; one summary CSV; optional per-spectrum **trace CSVs**; outputs go to your selected directory.

- **Persisted configuration**
  - Stored in `~/.gl_peakfit_config.json` (see **Persistence** above).

- **Unified exports**
  - `_fit.csv`, `_trace.csv`, `_uncertainty.csv`, `_uncertainty.txt` for **single** runs and **per-file** in **batch**; plus a batch **summary CSV**.

## Baseline methods
Peakfit now supports two baseline estimators:

**ALS (Asymmetric Least Squares)**  \
Smooth, robust baseline that penalizes points above the baseline. Tunables:
`lam` (smoothness), `p` (asymmetry), `niter` (iterations), `thresh` (optional stop threshold).

**Polynomial**  \
Weighted least-squares polynomial baseline on `(x, y)`. Tunables:
`degree` (non-negative integer), `normalize_x` (bool; if `true`, fits in scaled `[-1, 1]` to improve conditioning).
When fitting only a slice (fit range), the polynomial degree is automatically clamped to
`min(requested_degree, n_points_in_range - 1)`. The UI surfaces this adjustment.

### Choosing a method (GUI)
- In the **Baseline** panel, pick **Method: als** or **polynomial**.
- Use **Save as default** to persist the current method and its parameters for future sessions.
- The last used method is restored on next launch.

### Choosing a method (programmatic / batch)
Configure the run with:
```json
{
  "baseline": {
    "method": "als",
    "lam": 1e5,
    "p": 0.001,
    "niter": 10,
    "thresh": 0.0
  }
}
```
or
```json
{
  "baseline": {
    "method": "polynomial",
    "degree": 2,
    "normalize_x": true
  }
}
```
Set `baseline_uses_fit_range: true` to estimate the baseline only inside the fit window; otherwise the full trace is used.

### Exports
CSV peak tables include baseline metadata:
`baseline_method`, `als_*` fields (filled for ALS, `NaN` under polynomial),
and `poly_degree`, `poly_normalize_x` (filled for polynomial, `NaN` under ALS).

### Defaults & persistence
The app stores per-method defaults and the last used method under
`~/.gl_peakfit_config.json` in a `baseline_defaults` block:
```json
{
  "baseline_defaults": {
    "method": "polynomial",
    "als":  { "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0 },
    "polynomial": { "degree": 2, "normalize_x": true }
  }
}
```

---

## Install

> Python 3.10–3.12 recommended.

### Option A — conda (recommended on Windows)
```bash
conda create -n peakfit python=3.11 -y
conda activate peakfit
conda install numpy scipy pandas matplotlib -y
pip install lmfit emcee  # optional: lmfit for LMFIT-VP, emcee for Bayesian CIs
```
**Optional performance extras**
```bash
pip install numba                    # JIT on CPU
pip install "cupy-cuda11x<14"        # GPU (requires CUDA 11.x; 11.8 recommended)
```
> **Windows CUDA notes:** install **CUDA Toolkit 11.8** (for `cupy-cuda11x`). Set `CUDA_PATH` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` and ensure `...\bin` and `...\lib\x64` are on `PATH`.

### Option B — pip (virtualenv)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy scipy pandas matplotlib lmfit emcee
# optional
pip install numba "cupy-cuda11x<14"
```

---

## Run
```bash
python run_app.py
```
- Plot on the left, controls on the right; **Action bar** in the top-right.  
- **Help** (F1) is a scrollable window; right-panel scroll is independent.

---

## Quick Start

1. **Open Data…** (two-column x,y file).  
2. **Select a fit window** (drag or enter Min/Max).  
3. **Recompute baseline** (ALS); defaults are usually fine.  
4. **Add peaks** (click) or **Auto-seed**; lock Width/Center as needed; set η.  
5. Pick **Classic / Modern TRF / Modern VP / LMFIT-VP**.  
6. **Step ▶** to preview a single iteration, or **Fit**.  
7. **Uncertainty** → overlay band (asymptotic/boot/Bayesian).  
8. **Export** to produce `_fit.csv`, `_trace.csv`, `_uncertainty.csv`, `_uncertainty.txt`.  
9. **Batch** to process a folder; outputs go to your chosen directory.

---

## Uncertainty

- **Asymptotic**: covariance from Jacobian at the solution; 95% CI band via delta method (smoothed).  
- **Bootstrap**: residual resampling → refit → parameter stats and optional prediction band (seeded; respects locks/bounds).  
- **Bayesian**: MCMC (emcee) posterior mean/SD/95% CI; ESS/R-hat; predictive band. If `emcee` is not installed, the app reports **NotAvailable**.

---

## Export Schemas

### A) Peak Table `_fit.csv` (single & batch use the same columns and order)
```
file, peak, center, height, fwhm, eta, lock_width, lock_center,
area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax,
solver_choice, solver_loss, solver_weight, solver_fscale, solver_maxfev,
solver_restarts, solver_jitter_pct, step_lambda,
baseline_uses_fit_range, perf_numba, perf_gpu, perf_cache_baseline,
perf_seed_all, perf_max_workers
```
- `area`: pseudo-Voigt closed-form area  
- `area_pct`: 100 × area / Σ area  
- `rmse`: computed over the active fit window vs correct target (Add: raw y; Subtract: y − baseline)

### B) Trace `_trace.csv` (per spectrum; single & batch)
```
x, y_raw, baseline,
y_target_add, y_fit_add, peak1, peak2, …,
y_target_sub, y_fit_sub, peak1_sub, peak2_sub, …
```
- `peakN`     = baseline-**ADDED** component (matches Add-mode display)  
- `peakN_sub` = baseline-**SUBTRACTED** pure component (for calculations)

### C) Uncertainty
- `_uncertainty.csv`: param means, SDs, 95% CI; optional band summaries  
- `_uncertainty.txt`: human-readable report with **±** values and method notes

> No blank lines are written in any CSV.

---

## Batch Processing

- Select **folder & patterns** (semicolon-separated).  
- **Peaks source**: Current (optional **re-height**), **Template** (auto-applied), or **Auto-seed**.  
- Per-file baseline respects **“baseline uses fit range”** (default ON).  
- Writes per-file `_fit.csv`, `_trace.csv`, `_uncertainty.csv/.txt` **in your selected output folder**, plus a summary CSV.  
- Progress and messages stream to the status/log panel.

---

## Tips & Troubleshooting

- **Add vs Subtract**: In Add mode the baseline is added back into the model. If fits look too tall, double-check the mode.  
- **ALS tuning**: Increase λ (smoother) and/or decrease p (more under peaks) if ALS rides peak tops.  
- **Robustness**: For spikes/outliers, try TRF with `soft_l1`/`huber`/`cauchy` and Poisson/`inv_y` weights.  
- **Step ▶ rejected**: Try a smaller damping λ, refine seeds (centers/FWHM), or run a full Fit.  
- **Uncertainty “spikes”**: Lock weakly determined parameters, narrow the window, or increase bootstrap samples.  
- **CuPy warning** (`CUDA path could not be detected`): set `CUDA_PATH` to your Toolkit (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`) and add `...\bin;...\lib\x64` to `PATH`.  
- **GPU mismatch**: Ensure Toolkit **11.x** for `cupy-cuda11x`; newer toolkits may not match prebuilt wheels.

---

## Testing & Smoke Tools

- Run tests:
```bash
pytest -q
```
- Smoke utilities (paths may vary):
```bash
python tools/smoke_uncertainty.py tests/fixtures/real1.csv --method asymptotic
python tools/smoke_uncertainty.py tests/fixtures/real1.csv --method bootstrap
python tools/smoke_batch.py tests/fixtures --outdir ./.out
```

---

## Packaging (optional)

Create a single-file executable with PyInstaller:
```bash
pip install pyinstaller
pyinstaller -F -n PeakFit --add-data "ui;ui" --add-data "docs;docs" run_app.py
# macOS: consider -w; Windows: Tcl/Tk is typically auto-bundled.
```
The output binary will be in `dist/PeakFit` (or `PeakFit.exe` on Windows).

---

## Version History (high level)

- **v3.2-beta** – Stability + parity; Classic restored (`curve_fit`); unified Step ▶; settings persisted; action bar/log UI; **asymptotic/bootstrap/Bayesian** uncertainty; unified exports.  
- **v3.1** – Solver tuning: center-in-window (optional), Δx-based FWHM lower bound, param-wise jitter, x-scaling.  
- **v3.0** – Modern TRF with robust losses & weights; ALS iterations/threshold; scrollable Help; resizable panel; persisted x-label.  
- **v2.7** – Stable standalone release with ALS/Help/x-label persistence.

---

## License

MIT — see `LICENSE`.

## Citation

If this tool helps your research, please consider citing this repository:

```text
Zahin, F. (2025). Interactive Peak Fit GUI (pseudo-Voigt). GitHub repository.
```
