# Interactive Peak Fit GUI - Complete Codebase Documentation

## 1. Project Overview

**Interactive Peak Fit GUI** is a cross-platform Tkinter + Matplotlib desktop application for interactive spectral peak fitting with pseudo-Voigt (Gaussian-Lorentzian) peak models. 

**Core Purpose**: Enable researchers to fit spectral data with:
- Multiple peak fitting (up to 5+ peaks with automatic seeding)
- Baseline correction via ALS or polynomial methods
- Multiple solver backends (Classic, Modern TRF, Modern VP, optional LMFIT-VP)
- Robust uncertainty quantification (Asymptotic, Bootstrap, Bayesian/MCMC)
- Batch processing for high-throughput spectral analysis
- Comprehensive export formats (peak tables, traces, uncertainty estimates)

**Version**: v3.2-beta (19,345 lines of Python code across 214 files)

**Build Focus**: Reliability, parity between single and batch processing, unified fitting engine, and deterministic seeding for reproducibility.

---

## 2. Architecture

### High-Level Package Structure

```
Interactive Peak Fit GUI/
├── core/                 # Core fitting and data processing (1.1M)
│   ├── data_io.py       # File I/O, export formats, robustness utilities
│   ├── fit_api.py       # High-level fitting orchestration
│   ├── models.py        # Pseudo-Voigt peak mathematics
│   ├── peaks.py         # Peak data structures
│   ├── residuals.py     # Residual and Jacobian construction
│   ├── signals.py       # ALS baseline, SNR estimation, polynomial baseline
│   ├── uncertainty.py   # Uncertainty result class and routing (129K lines)
│   ├── uncertainty_router.py # Uncertainty method selection
│   ├── jacobians.py     # Pseudo-Voigt derivatives
│   ├── weights.py       # Noise weighting and robust loss schemes
│   ├── bounds.py        # Parameter bounds construction
│   └── mcmc_utils.py    # ESS/R-hat diagnostics for Bayesian
├── fit/                 # Solver implementations (332K)
│   ├── classic.py       # SciPy curve_fit backend (simple, lock-aware)
│   ├── modern.py        # Trust-Region-Reflective solver (robust losses/weights)
│   ├── modern_vp.py     # Variable-Projection solver (NNLS heights)
│   ├── lmfit_backend.py # Optional LMFIT-VP wrapper
│   ├── step_engine.py   # Single iteration stepping (shared contract)
│   ├── bounds.py        # Bound packing logic
│   ├── orchestrator.py  # Solver dispatch
│   ├── result.py        # FitResult dataclass
│   └── utils.py         # Helper utilities (MAD sigma, robust cost)
├── batch/               # Batch processing (191K)
│   ├── runner.py        # Batch orchestration (1,315 lines)
│   └── __init__.py
├── ui/                  # User Interface (1.2M)
│   ├── app.py          # Main Tkinter GUI (7,113 lines)
│   ├── helptext.py     # Help documentation
│   └── uncertainty_utils.py # UI uncertainty utilities
├── uncertainty/         # Legacy uncertainty modules (71K)
│   ├── asymptotic.py   # Asymptotic CI estimation
│   ├── bootstrap.py    # Bootstrap resampling
│   └── bayes.py        # Bayesian/MCMC with emcee
├── infra/               # Infrastructure (359K)
│   ├── config.py       # Configuration persistence (~/.gl_peakfit_config.json)
│   ├── performance.py  # Performance backend (Numba/CuPy, 577 lines)
│   └── logging.py      # Logging setup
├── tests/               # Comprehensive test suite (778K, 164 test files)
├── tools/               # Utility scripts
├── scripts/             # Additional scripts
├── docs/                # Documentation
└── requirements.txt     # Dependencies
```

### Dependency Graph

**Core Dependencies**:
- `numpy` - Numerical computation
- `scipy` - Optimization (curve_fit, least_squares, nnls)
- `pandas` - Data I/O and manipulation
- `matplotlib` - Plotting and visualization
- `tkinter` - GUI framework (built-in)

**Optional Dependencies**:
- `numba` - JIT compilation for performance
- `cupy` - GPU acceleration (CUDA-based)
- `lmfit` - Alternative solver backend
- `emcee>=3.1` - MCMC for Bayesian uncertainty
- `threadpoolctl` - Thread pool management

---

## 3. Core Functionality

### Peak Model
- **Pseudo-Voigt**: `g(x; h, c, w, η) = h * [(1-η)*G(x) + η*L(x)]`
  - `h` = height (≥ 0)
  - `c` = center (x-location)
  - `w` = FWHM (> 0)
  - `η` ∈ [0,1] = Gaussian-Lorentzian mix (0=pure Gaussian, 1=pure Lorentzian)
  - Closed-form area calculation: `area = (1-η)*gauss_area + η*lorentz_area`

### Fitting Modes
1. **Add Mode**: `y_fit = baseline + Σ(peaks)` fitted to raw `y`
2. **Subtract Mode**: `y_fit = Σ(peaks)` fitted to `(y - baseline)`

### Baseline Correction
Two methods supported:
1. **ALS (Asymmetric Least Squares)**
   - Parameters: `λ` (smoothness), `p` (asymmetry), `niter`, `thresh` (early stop)
   - Sparse tri-diagonal solver with iterative weight updates
   - Optional: compute only on fit window then interpolate

2. **Polynomial**
   - Parameters: `degree` (auto-clamped to data), `normalize_x` (scaling for stability)
   - Weighted least-squares via Vandermonde matrix

### Interactive Workflows
- **Click-to-add peaks**: Click on plot to add peaks at that location
- **Auto-seeding**: Detect prominent peaks via local maxima (configurable max 5)
- **Per-peak controls**: Lock center/width individually
- **Global η**: Apply shape factor uniformly to all peaks
- **Reheighting**: Refresh peak heights from current signal (batch feature)

---

## 4. Data Flow

### Single Fit Workflow
```
Load Data (CSV/TXT/DAT)
    ↓
Auto-detect delimiters, drop non-numeric, remove non-finite rows, sort x
    ↓
Compute Baseline (ALS or Polynomial on fit window or full range)
    ↓
Define Fit Window (min/max x range)
    ↓
Add Peaks (manually or auto-seed)
    ↓
Fit Selection (Classic / TRF / VP / LMFIT-VP)
    ↓
Run Solver → FitResult {θ, cost, RMSE, diagnostics}
    ↓
Compute Uncertainty (Asymptotic / Bootstrap / Bayesian)
    ↓
Compute Prediction Band (smoothed) if selected
    ↓
Plot overlay on data
    ↓
Export CSVs (_fit.csv, _trace.csv, _uncertainty.csv/.txt)
```

### Batch Workflow
```
Specify folder patterns (*.csv;*.txt;*.dat)
    ↓
Load config (solver, baseline, peaks source, uncertainty method)
    ↓
For each file matching patterns:
    Load data → Compute baseline → Seed peaks (current/template/auto)
    ↓
    [Optional] Re-height peaks from signal
    ↓
    Run fit with same engine as GUI
    ↓
    Compute uncertainty
    ↓
    Write per-file outputs (_fit.csv, _trace.csv, _unc.csv/.txt)
    ↓
Combine peak tables → batch summary CSV
```

---

## 5. Key Modules

### `core/models.py` (148 lines)
Pseudo-Voigt evaluation with optional GPU support:
- `pseudo_voigt(x, h, x0, w, η)` - Single peak evaluation
- `pv_sum(x, peaks)` - Sum of peaks
- `pv_sum_with_jac(x, centers, heights, fwhm, eta)` - Peak sum + Jacobian
- `pv_design_matrix(x, peaks)` - Design matrix for height-only fits
- `pv_area(h, w, η)` - Closed-form area

Supports both NumPy and CuPy via `xp` alias.

### `core/peaks.py` (29 lines)
Simple dataclass-based peak representation:
```python
@dataclass
class Peak:
    center: float
    height: float
    fwhm: float
    eta: float
    lock_center: bool = False
    lock_width: bool = False
```
Provides serialization/deserialization for JSON/CSV.

### `core/fit_api.py` (850+ lines)
High-level fitting orchestration:
- `run_fit_consistent()` - Main fit entry point with all solver choices
- Parameter packing/unpacking respecting locks and bounds
- Shared residual builder for all solvers
- Step engine dispatch
- Peak relabeling for multi-modal fits
- Prediction function builders for uncertainty bands

### `core/signals.py` (116 lines)
Baseline and signal utilities:
- `als_baseline(y, lam, p, niter, tol)` - ALS baseline estimation
- `polynomial_baseline(x, y, degree, normalize_x)` - Polynomial baseline
- `snr_estimate(y)` - Robust SNR via MAD

### `core/residuals.py` (184 lines)
Residual and Jacobian construction:
- `build_residual()` - Returns `r(θ)` function respecting mode/weights
- `build_residual_jac()` - Returns `(r(θ), J(θ))` for lock-aware packing
- `finite_diff_jacobian()` - Finite difference Jacobian estimation
- `jacobian_fd()` - NumPy/CuPy compatible FD Jacobian

### `core/weights.py` (50+ lines)
Noise weighting schemes:
- `none` - No weighting (pure LS)
- `poisson` - `w_i = 1/√|y_i|` (photon noise model)
- `inv_y` - `w_i = 1/|y_i|` (heteroscedastic noise)

### `core/uncertainty.py` (3,500+ lines)
Comprehensive uncertainty framework:
- `UncertaintyResult` - Canonical container with method/label/stats/band
- `asymptotic_ci()` - Jacobian-based covariance + delta method
- `bootstrap_ci()` - Residual resampling with optional prediction band
- `bayesian_ci()` - MCMC posterior (emcee) with ESS/R-hat diagnostics
- Band building, smoothing, and validation utilities
- Vectorized relabeling for peak identity consistency

### `fit/classic.py` (328 lines)
Simple SciPy `curve_fit` backend:
- Parameter packing with lock/bound support
- Minimal bounds: `h≥0`, `w≥eps`, optional center window clamp
- Direct interface to `scipy.optimize.curve_fit`
- No robust loss or weighting (by design—simple & fast)

### `fit/modern.py` (450+ lines)
Trust-Region-Reflective solver:
- `scipy.optimize.least_squares` with bounds enforcement
- Robust losses: `linear`, `soft_l1`, `huber`, `cauchy`
- Noise weighting: `none`, `poisson`, `inv_y`
- Parameter scaling (`x_scale`) for conditioning
- Optional restart and jitter mechanisms
- Analytic Jacobian when available, FD fallback

### `fit/modern_vp.py` (450+ lines)
Variable-Projection solver:
- Split parameters: linear (heights) ↔ nonlinear (centers/widths)
- Heights solved by NNLS: `argmin_{h≥0} ||W(A h - y)||₂`
- Centers/widths updated via damped Gauss-Newton + backtracking
- Robust loss and weighting support
- Step-halving and λ damping for stability

### `fit/step_engine.py` (250+ lines)
Single-iteration stepping (shared across solvers):
- Compute Jacobian at current θ
- Solve damped normal equations: `(J'J + λI)δ = -J'r`
- Evaluate cost at θ + δ
- Backtrack if cost not decreased
- Return diagnostics: `StepResult(accepted, λ, backtracks, step_norm, reason)`

### `batch/runner.py` (1,315 lines)
Batch processing engine:
- Glob file matching
- Per-file data loading with auto-baseline windowing
- Peak seeding (current/template/auto with optional re-height)
- Solver dispatch using same `core.fit_api` as GUI
- Uncertainty computation with multi-worker support
- Per-file and summary CSV export
- Progress streaming and abort handling

### `core/data_io.py` (2,500+ lines)
Robust I/O:
- Auto-detecting CSV/TXT/DAT readers (pandas with fallback)
- Delimiter auto-detection
- Comment stripping (`#`, `%`, `//`)
- Non-numeric/non-finite row filtering
- X-axis sorting
- Export builders:
  - `build_peak_table()` - Peak parameters + metadata
  - `build_trace_table()` - Full-domain traces (both Add/Subtract components)
  - Uncertainty CSV/TXT writers with ± notation and human-readable format
- Unified schemas for single and batch

### `infra/performance.py` (577 lines)
Defensive performance layer:
- Backend selection: `numpy` (default) → `numba` (JIT) → `cupy` (GPU)
- Safe fallback on backend unavailability or mismatch
- Shadow-compare mode for testing
- Peak evaluation: `eval_total()`, `eval_components()`
- Design matrix construction for NNLS
- Thread pool and BLAS control via `threadpoolctl`
- RNG seeding at module level for reproducibility

### `infra/config.py` (78 lines)
Persistent configuration management:
- File: `~/.gl_peakfit_config.json`
- Keys: baseline defaults, solver/options, uncertainty method, UI toggles, batch defaults, performance flags
- Load/save with recursive merge for backward compatibility

---

## 6. GUI/UI Structure

### Main Application (`ui/app.py`, 7,113 lines)

**Layout**:
- **Left pane**: Matplotlib canvas (plot)
- **Right pane**: Scrollable control panel with:
  - File/Baseline/Peaks/Solver/Uncertainty/Batch sections
  - Status bar at bottom (progress bar + collapsible log)
  - Action bar (top-right): File | Fit | Plot | Help (F1)

**Key UI Components**:
1. **Plot Canvas**
   - Shows raw data, baseline, fitted model, individual peaks
   - Clickable for peak addition (toggle mode)
   - Optional uncertainty band overlay

2. **Control Sections** (scrollable)
   - **Baseline**: ALS/Polynomial method selection, λ/p/degree tuning
   - **Peaks Panel**: Add/remove/lock peaks; per-peak η; global η broadcast
   - **Solver Selection**: Radio buttons for Classic/TRF/VP/LMFIT
   - **Solver Options**: Loss, weights, f_scale, maxfev, restarts, jitter
   - **Uncertainty**: Method (Asymptotic/Bootstrap/Bayesian), n_samples, band toggle
   - **Batch**: Patterns, source, output folder, progress

3. **Status & Logging**
   - Progress bar (percentage)
   - Collapsible green-on-black log panel
   - Real-time stream during batch/uncertainty operations

4. **Action Bar**
   - **File**: Open data, Export results, Batch processing
   - **Fit**: Step (one iteration), Fit (solve)
   - **Plot**: Show uncertainty band, Toggle legend, Show components
   - **Help**: F1 opens scrollable help window

**Persistence** (`~/.gl_peakfit_config.json`):
- Last solver choice and options
- Baseline method and defaults
- Uncertainty method and parameters
- UI toggles (legend visibility, performance flags)
- Global η, click-to-add toggle
- Batch defaults and templates
- X-axis label

---

## 7. Fitting System

### Solver Selection

#### Classic (SciPy `curve_fit`)
- **Use case**: Simple, fast, lock-aware, no outlier handling
- **Algorithm**: Levenberg-Marquardt
- **Bounds**: Minimal (h≥0, w≥eps, optional center window clamp)
- **Weighting**: None (pure LS)
- **Robust loss**: None
- **Locks**: Fully respected via parameter packing
- **Step ▶**: Uses same residual/bounds, performs 1 damped LM step

#### Modern TRF (SciPy `least_squares`)
- **Use case**: General purpose with robust loss handling
- **Algorithm**: Trust-Region-Reflective
- **Bounds**: Full support (h, c, w individually bounded)
- **Weighting**: Optional (poisson, inv_y)
- **Robust loss**: linear, soft_l1, huber, cauchy
- **Locks**: Respected via reduced theta packing
- **Conditioning**: Parameter scaling (x_scale)
- **Restarts**: Optional randomized jitter + re-solver
- **Step ▶**: Matches TRF objectives and bounds

#### Modern VP (Variable-Projection)
- **Use case**: Overlapping peaks, ill-conditioned center estimates
- **Algorithm**: NNLS (heights) + Damped Gauss-Newton (centers/widths)
- **Heights**: Solved exactly via NNLS (non-negative LS)
- **Centers/Widths**: Updated via reduced Gauss-Newton + backtracking
- **Bounds**: Enforced on nonlinear parameters; heights automatically ≥0
- **Locks**: Fully respected
- **Weighting**: Optional (same as TRF)
- **Robust loss**: Optional (same as TRF)
- **Step ▶**: NNLS + one damped GN iteration

#### LMFIT-VP (optional)
- **Wrapper** around lmfit's `minimize` for VP-like behavior
- Only available if `lmfit` is installed
- Reports `NotAvailable` if absent

### Parameter Packing

**Full theta** (always 4 × n_peaks):
```
θ_full = [c₁, h₁, w₁, η₁, c₂, h₂, w₂, η₂, ...]
```

**Packed theta** (respecting locks):
```
θ_packed = [h₁, (c₁), (w₁), h₂, (c₂), (w₂), ...]
            ↑   free unless locked
```
Heights always free; centers/widths omitted if locked.

### Bounds

**Classic**:
- Heights: [0, ∞]
- Widths: [eps_w, cap] (cap optional)
- Centers: [-∞, ∞] or [xmin, xmax] if `bound_centers_to_window`

**Modern/VP**:
- Heights: [0, max_height] (95th percentile × factor)
- Widths: [eps_w, max_fwhm] (0.5 × data span)
- Centers: [-∞, ∞] or [xmin, xmax] if configured

### Locks

Locked parameters are:
1. Removed from fitting vector
2. Held at their initial values
3. Respected across Classic, TRF, VP solvers
4. Preserved through bootstrap resampling
5. Honored during jitter (locked params not jittered)

---

## 8. Uncertainty Calculation

### Three Methods

#### Asymptotic (Default Band ON)
- **Algorithm**: Jacobian-based covariance + delta method
- **Computation**:
  1. Compute Jacobian J at solution θ*
  2. Estimate σ² = RSS / dof (residual variance)
  3. Cov = σ² (J'J)⁻¹ with Tikhonov regularization if needed
  4. Form gradient G = ∂ŷ/∂θ on full x (FD)
  5. Band variance: Var[Ŷ] = diag(G Cov G')
  6. Smoothed 95% CI: ŷ ± 1.96 √Var
- **Advantages**: Fast, always available, no hyperparameters
- **Limitations**: Assumes linear approximation, may underestimate with nonlinearity

#### Bootstrap (Residual Resampling)
- **Algorithm**: Resample residuals → refit → parameter statistics
- **Process**:
  1. Compute fitted residuals: r = y_target - y_fit
  2. Draw n_samples bootstrap residuals: r* = resample(r, replace=True)
  3. Synthetic data: y* = y_fit + r*
  4. Refit with same solver respecting locks/bounds
  5. Collect parameter draws → θ̂₁*, θ̂₂*, ..., θ̂ₙ*
  6. Compute mean, SD, 95% CI (percentiles)
  7. Optional: prediction band from ŷ* predictions
- **Determinism**: Seeded per-draw (seed_base + draw_index)
- **Parallelization**: Configurable worker pool (outer loop with BLAS limit)
- **Advantages**: Distribution-free, captures nonlinearity
- **Limitations**: Computationally expensive, discrete parameter draws needed for peak ID matching

#### Bayesian (MCMC, optional emcee)
- **Algorithm**: MCMC posterior sampling with ESS/R-hat diagnostics
- **Process**:
  1. Define log-likelihood: log π(θ|data) ∝ -0.5 RSS(θ)
  2. Define priors (flat with bounds)
  3. Run MCMC (emcee walkers) for burn-in + production
  4. Thin by autocorr; compute ESS, R̂ diagnostics
  5. Extract posterior mean/SD/95% credible intervals
  6. Optional: posterior predictive band
- **Diagnostics**: ESS (effective sample size), R̂ (potential scale reduction)
- **Advantages**: Posterior distributions, uncertainty on model assumptions
- **Limitations**: Requires emcee; MCMC tuning needed; slow for many parameters

### Band Construction

All methods can produce a **prediction band** `(x, y_lo, y_hi)`:
1. Collect parameter draws or posterior samples
2. For each θ draw: compute ŷ(x) on full domain
3. Stack predictions; compute 95% percentiles row-wise
4. Optional: smooth band with rolling median (window=5-7 points)
5. Gating: skip band if diagnostics unhealthy (low ESS, high R̂)

---

## 9. Batch Processing

### Configuration
```python
config = {
    "patterns": "*.csv;*.txt;*.dat",  # semicolon-separated globs
    "source": "template|current|auto",  # peak seeding
    "peaks": [...],                     # template peaks if source=='template'
    "reheight": False,                  # refresh heights per file
    "save_traces": True,                # per-file trace CSVs
    "auto_max": 5,                      # max auto-seeded peaks
    "compute_uncertainty": True,
    "unc_method": "asymptotic|bootstrap|bayesian",
    "solver": "modern_vp",
    "solver_loss": "linear",
    ...baseline, fit window, mode, etc...
}
```

### Workflow
1. **Glob files** matching patterns (case-insensitive on Windows)
2. **For each file**:
   - Load data
   - Compute baseline (ALS/poly) optionally within fit window
   - Seed peaks (reuse current/template with optional re-height, or auto)
   - Fit using `core.fit_api.run_fit_consistent()`
   - Compute uncertainty
   - Write per-file `_fit.csv`, `_trace.csv`, `_uncertainty.csv/.txt`
3. **Combine** all peak tables → batch summary CSV (one row per peak)
4. **Stream progress** and log messages back to UI

### Output Files
- **Summary**: `{prefix}_batch_summary.csv` (all peaks from all files)
- **Per-file**: 
  - `{filename}_fit.csv` (peak table)
  - `{filename}_trace.csv` (x, y, baseline, components)
  - `{filename}_uncertainty.csv` (param stats)
  - `{filename}_uncertainty.txt` (human-readable)

### Peak Seeding Strategies
1. **Current**: Use peaks from current GUI session (optional re-height from signal)
2. **Template**: Use saved template (auto-apply on startup if enabled)
3. **Auto**: Detect prominent local maxima (up to max_peaks)

---

## 10. Export/Import

### Import Formats
- **CSV/TXT/DAT**: Two-column (x, y) with auto-delimiter detection
- **Comments**: Lines starting with `#`, `%`, `//` are stripped
- **Headers**: Text columns are dropped; pure numeric pairs extracted
- **Cleaning**: Non-finite rows removed; x sorted ascending
- **Engine**: pandas (with NumPy fallback)

### Export Formats

#### A) Peak Table (`*_fit.csv`)
Fixed column order (single & batch identical):
```
file, peak, center, height, fwhm, eta, lock_width, lock_center,
area, area_pct, rmse, fit_ok, mode,
als_lam, als_p, fit_xmin, fit_xmax,
solver_choice, solver_loss, solver_weight, solver_fscale, solver_maxfev,
solver_restarts, solver_jitter_pct, step_lambda,
baseline_uses_fit_range, perf_numba, perf_gpu, perf_cache_baseline,
perf_seed_all, perf_max_workers
```
- `area`: Closed-form pseudo-Voigt area
- `area_pct`: Normalized to sum=100%
- `rmse`: Computed on active fit window
- Baseline metadata: method, ALS params (λ, p) or poly degree

#### B) Trace CSV (`*_trace.csv`)
Full-domain traces (both Add and Subtract modes):
```
x, y_raw, baseline,
y_target_add, y_fit_add, peak1, peak2, ...,
y_target_sub, y_fit_sub, peak1_sub, peak2_sub, ...
```
- `peakN`: Baseline-ADDED component (display mode)
- `peakN_sub`: Baseline-SUBTRACTED pure component (calculation)
- No blank rows

#### C) Uncertainty CSV (`*_uncertainty.csv`)
Parameter statistics (wide and tall formats):
```
param | mean | sd | p2_5 | p97_5 | ...
```
Optional: band summaries if prediction band computed

#### D) Uncertainty TXT (`*_uncertainty.txt`)
Human-readable with ± notation:
```
Method: Asymptotic
Label: Asymptotic 95% CI

Center [param]:     123.45 ± 0.12
Height [param]:     456.78 ± 2.34
...
```

---

## 11. Performance/Optimization

### Backends
- **NumPy**: Default, always available
- **Numba**: Optional JIT compilation via `@numba.jit`
- **CuPy**: Optional GPU via CUDA (requires matching version)

### Defensive Layer (`infra/performance.py`)
1. **Graceful fallback**: If GPU/Numba unavailable, silently use NumPy
2. **Shadow compare**: Cross-check fast path against NumPy reference (test mode)
3. **Backend switching**: `set_numba()`, `set_gpu()` at runtime
4. **Warm-up**: Initialize Numba/GPU on first activation

### Optimization Strategies
- **NNLS for heights**: Avoids ill-conditioning of full parameter space
- **Variable projection**: Reduces effective dimensionality
- **Parameter scaling** (`x_scale`): Improves numerical conditioning in TRF
- **Sparse baseline operators**: O(n) tri-diagonal solve for ALS
- **Baseline caching**: Optional cache of polynomial baseline across repeated evaluations
- **BLAS limiting**: Cap threads per BLAS operation when running parallel batch workers

### Threading & Multiprocessing
- **Batch**: ProcessPoolExecutor for uncertainty (bootstrap/Bayesian draws)
- **Worker cap**: Configurable `perf_max_workers` (default: CPU count - 1)
- **Strategy**: "outer" (parallelize batch files), "inner" (parallelize bootstrap draws)
- **BLAS control**: `threadpool_limits` from `threadpoolctl` to prevent oversubscription

### Seeding & Reproducibility
- **Seed control**: `perf_seed_all` flag (global RNG seeding across NumPy/CuPy/random)
- **Per-draw seeding**: Bootstrap uses `seed_base + draw_index` for determinism
- **Bootstrap jitter**: Seeded jitter respects locks and doesn't perturb locked params

---

## 12. Testing

### Structure (164 test files, 778K)
Test categories:
1. **Solver tests**: Verify Classic/TRF/VP/LMFIT correctness and parity
2. **Baseline tests**: ALS, polynomial, fit-range behavior
3. **Uncertainty tests**: Asymptotic/bootstrap/Bayesian reproducibility
4. **Export tests**: CSV schema, no blank lines, header order
5. **Batch tests**: Single-batch parity, per-file outputs, progress streaming
6. **Performance tests**: Backend switching, Numba/GPU fallback, caching
7. **UI tests**: Config persistence, peak list updates, export dialogs
8. **Integration tests**: Full workflows (load → fit → export)

### Key Test Fixtures
- `two_peak_data`: Synthetic 2-peak spectrum
- `real1.csv`, `real2.csv`: Real spectral data
- `batch_small/`: Sample files for batch testing

### Testing Utilities
- `conftest.py`: Pytest configuration, RNG fixture, two_peak_data fixture
- `tools/smoke_uncertainty.py`: Manual uncertainty verification
- `tools/smoke_batch.py`: Manual batch processing test

---

## 13. Configuration

### Persistent Config File
**Location**: `~/.gl_peakfit_config.json`

**Structure**:
```json
{
  "version": "3.0",
  "baseline_defaults": {
    "method": "polynomial|als",
    "als": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
    "polynomial": {"degree": 2, "normalize_x": true}
  },
  "solver": "modern_vp|classic|modern_trf|lmfit_vp",
  "solver_loss": "linear",
  "solver_weights": "none|poisson|inv_y",
  "solver_fscale": 1.0,
  "solver_maxfev": 20000,
  "solver_restarts": 1,
  "solver_jitter_pct": 0.0,
  "step_lambda": 0.001,
  "uncertainty_method": "asymptotic|bootstrap|bayesian",
  "uncertainty_n_samples": 1000,
  "show_uncertainty_band": true,
  "add_peaks_on_click": true,
  "global_eta": 0.5,
  "show_legend": true,
  "perf_numba": false,
  "perf_gpu": false,
  "perf_cache_baseline": true,
  "perf_seed_all": true,
  "perf_max_workers": 4,
  "batch": {
    "patterns": "*.csv;*.txt;*.dat",
    "source": "template|current|auto",
    "reheight": false,
    "save_traces": true,
    "auto_max": 5
  },
  "x_label": "Wavenumber (cm⁻¹)",
  "templates": [
    {"name": "template_name", "peaks": [...], "auto_apply": true}
  ]
}
```

### Load/Save Mechanism
- Loads on app startup via `infra.config.load()`
- Saves on GUI changes via `save_config()`
- Recursive merge for backward compatibility
- Key migrations for legacy field names

---

## 14. Dependencies

### Core Dependencies
| Package | Purpose | Version |
|---------|---------|---------|
| `numpy` | Numerical arrays, linear algebra | Latest |
| `scipy` | `curve_fit`, `least_squares`, `nnls` optimization | >=1.7 |
| `pandas` | Data I/O, CSV manipulation | >=1.0 |
| `matplotlib` | Plotting, interactive canvas | >=3.0 |

### Optional Performance
| Package | Purpose | Impact |
|---------|---------|--------|
| `numba` | JIT compilation (CPU) | 2-5× speedup on peak eval |
| `cupy-cuda1Xx` | GPU acceleration | 10-100× speedup (if GPU available) |

### Optional Features
| Package | Purpose |
|---------|---------|
| `lmfit` | Alternative solver backend (LMFIT-VP) |
| `emcee>=3.1` | Bayesian MCMC uncertainty |
| `threadpoolctl` | BLAS thread pool management |

### Development/Testing
| Package | Purpose |
|---------|---------|
| `pytest` | Test framework |
| `pyinstaller` | Binary packaging |

---

## 15. Important Design Patterns

### 1. **Lock-Aware Parameter Packing**
Pattern: Free parameters packed into reduced vector θ, locked parameters held fixed.
- Benefit: Ensures locked params never drift
- Used in: All solvers, bootstrap resampling, jitter
- Implementation: Struct with index mappings (ih, ic, iw) in Classic; free_params list in TRF/VP

### 2. **Dual Residual/Jacobian Functions**
Pattern: Build both `r(θ)` and `(r(θ), J(θ))` to support finite-diff and analytic paths.
- Benefit: Solvers can choose; fallback to FD if analytic unavailable
- Used in: `core/residuals.py`, all solvers
- Implementation: `build_residual()` and `build_residual_jac()`

### 3. **Solver Dispatch via Options Dict**
Pattern: Options dict specifies solver; backends implement common interface.
- Benefit: Unified `solve(x, y, peaks, mode, baseline, options)` for all solvers
- Used in: `fit/__init__.py` dispatcher, batch runner
- Implementation: `options['solver']` maps to backend function

### 4. **Mode Isolation (Add vs Subtract)**
Pattern: Dual-mode residuals computed inline (no separate models).
- Benefit: Single fitting engine handles both modes
- Add: `r = (baseline + model) - y_raw`
- Subtract: `r = model - (y_raw - baseline)`
- Used in: All residual builders, step engine, exports

### 5. **Uncertainty Result Container**
Pattern: `UncertaintyResult` with canonical fields (method, label, stats, band, diagnostics).
- Benefit: Unified interface across Asymptotic/Bootstrap/Bayesian
- Backward compat: Dict-like `__getitem__` access
- Used in: UI display, export, batch summary

### 6. **Defensive Backend Switching**
Pattern: Try fast path (Numba/GPU) with immediate fallback to NumPy on error.
- Benefit: Graceful degradation; users don't notice if GPU unavailable
- Used in: `infra/performance.py`, peak evaluation
- Implementation: Try-except with `_BACKEND` flag reset

### 7. **Shared Step Engine**
Pattern: Single-iteration stepping uses same residuals/weights/bounds as full Fit.
- Benefit: Step ▶ visually "walks" the same landscape
- Used in: All solvers via `step_engine.py`
- Implementation: `iterate(state)` mirrors `solve()` objectives

### 8. **Deterministic Seeding**
Pattern: RNG seed can be fixed globally or per-draw (seed_base + index).
- Benefit: Reproducible uncertainty estimates, batch parity
- Used in: Bootstrap, Bayesian, jitter, batch processing
- Implementation: `perf_seed_all` flag, per-draw seed = base_seed + draw_idx

### 9. **Baseline Windowing**
Pattern: Baseline optionally computed only on fit window, then interpolated to full x.
- Benefit: Improves local baseline fit, prevents rippling at edges
- Flag: `baseline_uses_fit_range` (default ON)
- Used in: Single fit, batch, all methods

### 10. **CSV Schema Unification**
Pattern: Single and batch export to same column order (no variations).
- Benefit: Downstream tooling needs one parser
- Used in: Single export, batch per-file export, batch summary
- Implementation: `build_peak_table()`, `build_trace_table()`

---

## 16. Known Limitations

### Mathematical
1. **Pseudo-Voigt only**: No Lorentzian-squared or other peak shapes
2. **Single baseline**: Only one baseline across all peaks (no per-peak baselines)
3. **Fixed η**: Lorentzian fraction not optimized (user-set only)
4. **Linear fit window**: No polynomial/spline window boundaries (rect only)

### Solver
1. **Classic**: No robust loss or noise weighting (trade-off for simplicity)
2. **VP**: NNLS requires non-negative heights (can fail on inverted peaks)
3. **Bounds stiffness**: Tight bounds near limits can cause solver stalling
4. **Restart jitter**: Random jitter may not escape all local minima

### Uncertainty
1. **Asymptotic**: Assumes linearity; underestimates with strong nonlinearity
2. **Bootstrap**: Assumes residuals are exchangeable (may fail with structure/trends)
3. **Bayesian**: Requires emcee installation; MCMC tuning needed for complex posteriors
4. **Band smoothing**: Smoothing can mask fine structure in uncertainty

### GUI
1. **Large datasets**: Plotting 100k+ points can be slow (consider decimation)
2. **Interactive editing**: Dragging peaks on plot not supported (manual entry only)
3. **Multi-file comparison**: No side-by-side view for spectra
4. **Result caching**: No undo/redo history

### Batch
1. **File size limit**: Very large files may exhaust memory (no streaming I/O)
2. **Pattern matching**: Glob patterns are case-sensitive on Unix
3. **Partial failure**: Failure in one file doesn't skip remaining files

---

## 17. File Structure with Descriptions

```
Interactive Peak Fit GUI/
│
├── core/                             Core fitting engine
│   ├── __init__.py                   Package exports
│   ├── data_io.py (2,500 lines)      CSV I/O, export builders
│   ├── fit_api.py (850 lines)        High-level fit orchestration
│   ├── models.py (148 lines)         Pseudo-Voigt peak math
│   ├── peaks.py (29 lines)           Peak dataclass
│   ├── residuals.py (184 lines)      Residual/Jacobian builders
│   ├── signals.py (116 lines)        ALS, polynomial baseline, SNR
│   ├── uncertainty.py (3,500 lines)  Uncertainty result + internals
│   ├── uncertainty_router.py (200)   Uncertainty method selection
│   ├── jacobians.py (50 lines)       Pseudo-Voigt derivatives
│   ├── weights.py (50 lines)         Noise weighting schemes
│   ├── bounds.py (150 lines)         Bound construction
│   ├── mcmc_utils.py (50 lines)      ESS/R-hat diagnostics
│   └── robust.py (10 lines)          Robust loss functions
│
├── fit/                              Solver backends
│   ├── __init__.py                   Dispatcher interface
│   ├── classic.py (328 lines)        SciPy curve_fit backend
│   ├── modern.py (450+ lines)        TRF solver (robust losses)
│   ├── modern_vp.py (450+ lines)     Variable-Projection solver
│   ├── lmfit_backend.py (250 lines)  Optional LMFIT wrapper
│   ├── step_engine.py (250 lines)    Single-iteration stepping
│   ├── bounds.py (100 lines)         Bound packing for TRF/VP
│   ├── orchestrator.py (50 lines)    Solver dispatch
│   ├── result.py (30 lines)          FitResult dataclass
│   └── utils.py (50 lines)           MAD sigma, robust cost
│
├── uncertainty/                      Legacy uncertainty modules
│   ├── __init__.py
│   ├── asymptotic.py (58 lines)      Jacobian-based covariance
│   ├── bootstrap.py (183 lines)      Residual resampling
│   └── bayes.py (200 lines)          MCMC + diagnostics
│
├── batch/                            Batch processing
│   ├── __init__.py
│   └── runner.py (1,315 lines)       Main batch orchestrator
│
├── ui/                               GUI
│   ├── __init__.py
│   ├── app.py (7,113 lines)          Main Tkinter application
│   ├── helptext.py (15k lines)       Help documentation
│   └── uncertainty_utils.py (100)    UI uncertainty utilities
│
├── infra/                            Infrastructure
│   ├── __init__.py
│   ├── config.py (78 lines)          Config file handling
│   ├── performance.py (577 lines)    Backend switching (Numba/GPU)
│   └── logging.py (50 lines)         Logging setup
│
├── tests/                            Test suite (164 files)
│   ├── conftest.py                   Pytest fixtures
│   ├── fixtures/                     Test data
│   │   ├── real1.csv                 Real spectrum 1
│   │   ├── real2.csv                 Real spectrum 2
│   │   └── batch_small/              Batch test files
│   ├── test_*.py                     Individual test files
│   └── test_z_audit.py               Regression tests
│
├── tools/                            Utility scripts
│   ├── __init__.py
│   ├── smoke_uncertainty.py          Manual uncertainty test
│   ├── smoke_batch.py                Manual batch test
│   ├── batch_cli.py                  CLI batch interface
│   └── _run_one_batch.py             Single batch job runner
│
├── scripts/                          Additional scripts
├── docs/                             Documentation
├── run_app.py                        App launcher (thin wrapper)
├── gl_decon_gui_batch.py             Batch CLI entry point
├── requirements.txt                  Dependencies
├── README.md                         User guide
└── LICENSE                           MIT license
```

---

## Summary

The Interactive Peak Fit GUI is a comprehensive, production-ready spectral analysis tool that demonstrates:

1. **Modular design**: Clear separation between core (fitting), UI, batch, and infrastructure
2. **Multiple solver strategies**: Classic (simple) → TRF (robust) → VP (ill-conditioned)
3. **Sophisticated uncertainty**: Three complementary methods (asymptotic, bootstrap, Bayesian)
4. **Reproducibility**: Deterministic seeding, single-batch parity, comprehensive tests
5. **Performance flexibility**: Optional GPU/Numba backends with safe fallbacks
6. **User-friendly**: Interactive GUI with real-time feedback, batch processing, detailed exports
7. **Robustness**: Defensive programming, extensive test coverage (164 files), lock/bound enforcement

The codebase (19,345 lines) balances feature richness with maintainability through clear abstractions and comprehensive testing.
