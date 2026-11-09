# Interactive Peak Fit GUI - Comprehensive Codebase Analysis

## Executive Summary

**Interactive Peak Fit GUI** is a production-ready cross-platform Tkinter + Matplotlib desktop application for interactive spectral peak fitting using pseudo-Voigt (Gaussian-Lorentzian mixture) peak models. 

**Project Statistics:**
- **Total Code**: 29,614 lines of Python
- **Core Modules**: 19 files (8,398 lines)
- **Solver Implementations**: 11 files (2,199 lines)
- **GUI Application**: 1 main file (7,113 lines)
- **Batch Processing**: 1 orchestrator (1,315 lines)
- **Test Suite**: 163 test files comprehensive coverage
- **Version**: v3.2-beta
- **License**: MIT
- **Author**: Farhan Zahin (designed), ChatGPT (built with)

---

## 1. Project Purpose & Overview

### Core Functionality
The application enables researchers to:
- **Load spectral data** from CSV/TXT/DAT files (2-column x,y format)
- **Interactively fit multiple peaks** using pseudo-Voigt model with configurable Gaussian-Lorentzian mixing
- **Correct baselines** using ALS (Asymmetric Least Squares) or polynomial methods
- **Select from multiple solvers**: Classic (curve_fit), Modern TRF (Trust-Region-Reflective), Modern VP (Variable-Projection), optional LMFIT
- **Quantify uncertainty** via three complementary methods: Asymptotic (covariance-based), Bootstrap (residual resampling), Bayesian (MCMC)
- **Process batches** of files with consistent configuration and deterministic seeding
- **Export results** in unified formats: peak tables, traces, uncertainty estimates with human-readable ± notation

### Key Features
1. **Interactive GUI** - Real-time plot updating, click-to-add peaks, lock/unlock parameters
2. **Robust Fitting** - Handles outliers via soft_l1, huber, and cauchy robust loss functions
3. **Flexible Baseline** - ALS with windowing or polynomial with automatic degree clamping
4. **Reproducibility** - Deterministic seeding, lock-aware jitter, per-draw seed control
5. **Performance** - Optional GPU (CuPy) and JIT (Numba) acceleration with safe fallback
6. **Persistence** - Configuration saved to ui/ui_config/gl_peakfit_config.json (beside the code, synced by OneDrive) with backward-compatible merging
7. **Comprehensive Testing** - 163 test files covering solvers, uncertainty, batch processing, exports
8. **Batch Processing** - Process folders with pattern matching, template seeding, progress streaming

---

## 2. Project Organization & Directory Structure

```
Interactive Peak Fit GUI/
│
├── core/                           # Core fitting engine (8,398 lines)
│   ├── __init__.py
│   ├── peaks.py                    # Peak dataclass (29 lines)
│   ├── models.py                   # Pseudo-Voigt mathematics (148 lines)
│   ├── fit_api.py                  # Fit orchestration (783 lines)
│   ├── fit_context.py              # Typed fitting context (242 lines)
│   ├── fit_orchestrator.py         # Baseline/peak/solver composition (315 lines)
│   ├── residuals.py                # Residual/Jacobian builders (184 lines)
│   ├── signals.py                  # ALS, polynomial baseline (116 lines)
│   ├── data_io.py                  # CSV I/O, export builders (2,219 lines)
│   ├── uncertainty.py              # Uncertainty computation (3,217 lines)
│   ├── uncertainty_router.py       # Method selection (271 lines)
│   ├── weights.py                  # Noise weighting schemes (70 lines)
│   ├── bounds.py                   # Parameter bounds (175 lines)
│   ├── bounds_classic.py           # Classic solver bounds (58 lines)
│   ├── jacobians.py                # Pseudo-Voigt derivatives (39 lines)
│   ├── param_utils.py              # Parameter packing utilities (202 lines)
│   ├── array_utils.py              # Array operations (273 lines)
│   ├── mcmc_utils.py               # ESS/R-hat diagnostics (38 lines)
│   ├── robust.py                   # Robust loss functions (8 lines)
│   └── models.py
│
├── fit/                            # Solver backends (2,199 lines)
│   ├── __init__.py                 # Dispatcher interface (134 lines)
│   ├── base.py                     # Abstract solver base (277 lines)
│   ├── classic.py                  # SciPy curve_fit (328 lines)
│   ├── modern.py                   # TRF solver (292 lines)
│   ├── modern_vp.py                # Variable-Projection (443 lines)
│   ├── lmfit_backend.py            # Optional LMFIT wrapper (233 lines)
│   ├── step_engine.py              # Single-iteration stepping (243 lines)
│   ├── bounds.py                   # Bound packing (157 lines)
│   ├── orchestrator.py             # Solver dispatch (35 lines)
│   ├── result.py                   # FitResult dataclass (26 lines)
│   └── utils.py                    # Helpers (31 lines)
│
├── batch/                          # Batch processing
│   ├── __init__.py
│   └── runner.py                   # Main orchestrator (1,315 lines)
│
├── ui/                             # User Interface
│   ├── __init__.py
│   ├── app.py                      # Main Tkinter application (7,113 lines)
│   ├── helptext.py                 # Help documentation (~15k lines)
│   └── uncertainty_utils.py        # UI uncertainty helpers (100 lines)
│
├── uncertainty/                    # Legacy uncertainty modules (71K total)
│   ├── __init__.py
│   ├── asymptotic.py               # Asymptotic CI
│   ├── bootstrap.py                # Bootstrap resampling
│   └── bayes.py                    # Bayesian/MCMC
│
├── infra/                          # Infrastructure
│   ├── __init__.py
│   ├── config.py                   # Config persistence (78 lines)
│   ├── performance.py              # Backend switching (577 lines)
│   └── logging.py                  # Logging setup (50 lines)
│
├── tests/                          # Test suite (163 files, comprehensive)
│   ├── conftest.py                 # Pytest configuration
│   ├── fixtures/                   # Test data
│   │   ├── real1.csv               # Real spectrum 1
│   │   ├── real2.csv               # Real spectrum 2
│   │   └── batch_small/            # Batch test files
│   └── test_*.py                   # Individual test files
│
├── tools/                          # Utility scripts
│   ├── __init__.py
│   ├── smoke_uncertainty.py        # Manual uncertainty test
│   ├── smoke_batch.py              # Manual batch test
│   ├── batch_cli.py                # CLI batch interface
│   └── _run_one_batch.py           # Single batch job
│
├── scripts/                        # Additional scripts
├── docs/                           # Documentation
├── run_app.py                      # Thin launcher wrapper (23 lines)
├── gl_decon_gui_batch.py           # Batch CLI entry point
├── requirements.txt                # Dependencies
├── README.md                       # User guide (302 lines)
├── CODEBASE_KNOWLEDGE.md           # Developer documentation (929 lines)
├── REFACTORING_SUMMARY.md          # Recent refactoring work (325 lines)
├── .gitignore
└── LICENSE                         # MIT license
```

---

## 3. Core Technologies & Dependencies

### Primary Dependencies
| Package | Purpose | Version |
|---------|---------|---------|
| `numpy` | Numerical computation, arrays | Latest |
| `scipy` | Optimization (curve_fit, least_squares, nnls) | >=1.7 |
| `pandas` | CSV/data manipulation | >=1.0 |
| `matplotlib` | Plotting, interactive canvas | >=3.0 |
| `tkinter` | GUI framework (built-in) | - |

### Optional Performance
| Package | Purpose | Impact |
|---------|---------|--------|
| `numba` | JIT compilation (CPU) | 2-5× speedup |
| `cupy-cuda1Xx` | GPU acceleration (CUDA 11.x) | 10-100× speedup |

### Optional Features
| Package | Purpose |
|---------|---------|
| `lmfit` | Alternative solver (LMFIT-VP) |
| `emcee>=3.1` | Bayesian MCMC uncertainty |
| `threadpoolctl` | BLAS thread management |

### Development/Testing
| Package | Purpose |
|---------|---------|
| `pytest` | Test framework |
| `pyinstaller` | Binary packaging |

---

## 4. Mathematical Foundation

### Pseudo-Voigt Peak Model

Each peak is modeled as a weighted combination of Gaussian and Lorentzian functions:

```
g(x; h, c, w, η) = h * [(1-η) * G(x) + η * L(x)]
```

Where:
- **h** ≥ 0: Peak height (amplitude)
- **c**: Peak center (x-location)
- **w** > 0: Full-Width Half-Maximum (FWHM)
- **η** ∈ [0,1]: Lorentzian-Gaussian mixing factor
  - η = 0: Pure Gaussian
  - η = 1: Pure Lorentzian
  - 0 < η < 1: Mixed pseudo-Voigt

Gaussian component:
```
G(x) = exp(-4 ln(2) * ((x-c)/w)²)
```

Lorentzian component:
```
L(x) = 1 / (1 + 4 * ((x-c)/w)²)
```

Closed-form area calculation:
```
Area = (1-η) * h * w * √(π/(4·ln(2))) + η * h * w * π/2
```

### Fitting Models

**Add Mode:**
```
y_fit = baseline + Σⱼ g_j(x; h_j, c_j, w_j, η_j)
fitted to raw y
residuals: r = y_fit - y_raw
```

**Subtract Mode:**
```
y_fit = Σⱼ g_j(x; h_j, c_j, w_j, η_j)
fitted to (y - baseline)
residuals: r = y_fit - (y_raw - baseline)
```

---

## 5. Main Components & Responsibilities

### 5.1 Core Module (`core/`)

#### `models.py` - Peak Mathematics (148 lines)
- **Purpose**: Pure mathematical evaluation of pseudo-Voigt peaks
- **Key Functions**:
  - `pseudo_voigt(x, h, c, w, η)`: Single peak evaluation
  - `pv_sum(x, peaks)`: Sum of multiple peaks
  - `pv_sum_with_jac(x, centers, heights, fwhm, eta)`: Peaks + analytic Jacobian
  - `pv_design_matrix(x, peaks)`: Design matrix for height-only fits (NNLS)
  - `pv_area(h, w, η)`: Closed-form area
- **Backend Support**: NumPy and CuPy (GPU) via `xp` alias
- **Design**: Vectorized for performance, no solver-specific logic

#### `peaks.py` - Peak Container (29 lines)
- **Purpose**: Simple data structure for peak representation
- **Core Class**:
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
- **Functions**: `serialize()` and `deserialize()` for JSON/CSV round-tripping

#### `fit_api.py` - High-Level Fitting Orchestration (783 lines)
- **Purpose**: Central fitting engine orchestrating all components
- **Main Function**: `run_fit_consistent()`
  - Accepts raw data, peaks, configuration, and fitting parameters
  - Handles baseline computation, peak initialization, solver selection
  - Returns comprehensive fit results with optional Jacobian/predictors
  - Respects locks, bounds, and constraints across all solvers
- **Key Helpers**:
  - Parameter packing/unpacking with lock awareness
  - Step engine dispatch for single-iteration stepping
  - Peak relabeling for multi-modal fits
  - Prediction function builders
- **Design**: Unified interface for Classic, TRF, VP, LMFIT solvers

#### `signals.py` - Baseline & SNR (116 lines)
- **Purpose**: Baseline estimation and signal analysis
- **Key Functions**:
  - `als_baseline(y, lam, p, niter, tol)`: ALS algorithm with sparse solver
  - `polynomial_baseline(x, y, degree, normalize_x)`: Polynomial fit
  - `snr_estimate(y)`: Robust SNR via MAD (Median Absolute Deviation)
- **Design**: Independent baseline methods, both support windowing

#### `residuals.py` - Residual & Jacobian Construction (184 lines)
- **Purpose**: Build residual and Jacobian functions for optimization
- **Key Functions**:
  - `build_residual()`: Returns r(θ) respecting mode/weights/locks
  - `build_residual_jac()`: Returns (r(θ), J(θ)) pair for analytic gradient
  - `finite_diff_jacobian()`: NumPy/CuPy-compatible FD estimation
  - `jacobian_fd()`: Row-wise FD computation
- **Design**: Supports both analytic and finite-difference paths, mode-agnostic

#### `weights.py` - Noise Weighting (70 lines)
- **Purpose**: Implement robust noise weighting schemes
- **Schemes**:
  - `none`: Pure least squares (w_i = 1)
  - `poisson`: Shot noise model (w_i = 1/√|y_i|)
  - `inv_y`: Heteroscedastic noise (w_i = 1/|y_i|)
- **Design**: Modular, can be extended for custom noise models

#### `bounds.py` - Parameter Bounds (175 lines)
- **Purpose**: Construct and manage parameter bounds for solvers
- **Strategies**:
  - **Classic**: Minimal bounds (h≥0, w>eps, optional center window)
  - **Modern**: Full bounds with data-driven limits
  - **VP**: Nonlinear parameter bounds + NNLS constraints
- **Design**: Respects locks (locked params removed from optimization)

#### `data_io.py` - CSV I/O & Export (2,219 lines)
- **Purpose**: Robust data loading and result export
- **Data Loading**:
  - Auto-detect delimiters (,;\\t| etc)
  - Strip comments (#, %, //)
  - Drop non-numeric columns
  - Remove non-finite rows
  - Sort x ascending if needed
  - Pandas-based with NumPy fallback
- **Export Builders**:
  - `build_peak_table()`: Peak parameters with full metadata
  - `build_trace_table()`: Full-domain traces (Add/Subtract modes)
  - `write_uncertainty_csv()`: Parameter stats (means, SDs, CIs)
  - `write_uncertainty_txt()`: Human-readable with ± notation
- **Design**: Unified schemas for single and batch processing, no blank lines

#### `uncertainty.py` - Uncertainty Quantification (3,217 lines)
- **Purpose**: Core uncertainty computation engine
- **Methods**:
  - **Asymptotic**: Jacobian-based covariance + delta method
  - **Bootstrap**: Residual resampling with seeded determinism
  - **Bayesian**: MCMC posterior (emcee) with ESS/R-hat diagnostics
- **Output**:
  - `UncertaintyResult`: Canonical container with stats/bands/diagnostics
  - Parameter means, SDs, 95% credible intervals
  - Optional prediction bands (smoothed with rolling median)
- **Design**: Vectorized, supports parallel workers, comprehensive diagnostics

#### `uncertainty_router.py` - Method Selection (271 lines)
- **Purpose**: Route uncertainty computation to appropriate backend
- **Functionality**:
  - Parse method string (asymptotic/bootstrap/bayesian)
  - Validate parameters
  - Check optional dependencies (emcee)
  - Return suitable handler
- **Design**: Clean abstraction for method dispatch

#### `param_utils.py` - Parameter Utilities (202 lines)
- **Purpose**: Centralized parameter packing/unpacking logic (refactored)
- **Key Functions**:
  - `vp_unpack_centers_widths()`: Extract nonlinear params from vector
  - `create_parameter_unpacker()`: Factory for reusable unpackers
  - `pack_nonlinear_params()`: Inverse operation
  - `validate_param_shapes()`: Shape validation
- **Design**: DRY principle, reduces duplication across solvers

#### `array_utils.py` - Array Operations (273 lines)
- **Purpose**: Safe, validated array operations
- **Functions**:
  - `ensure_1d_array()`: Convert to 1D with type coercion
  - `match_shapes()`: Validate array shapes
  - `broadcast_to_shape()`: Safe broadcasting
  - `stack_columns()`: Column stacking with validation
- **Design**: Explicit error messages, type hints for early detection

#### `fit_context.py` - Typed Fitting Context (242 lines)
- **Purpose**: Type-safe fitting state container (replaces dict)
- **Features**:
  - 22 typed attributes with defaults
  - Dictionary-like access for backward compatibility
  - Validation method for consistency checks
  - `to_dict()` / `from_dict()` serialization
- **Design**: Improves type safety, IDE autocomplete, reduces defensive checks

#### `fit_orchestrator.py` - Workflow Composition (315 lines)
- **Purpose**: Modularize baseline/peak/solver setup
- **Components**:
  - `BaselineConfig`: Configuration container
  - `BaselineCompute`: ALS/polynomial with windowing
  - `PeakInitializer`: Reheighting logic
  - `SolverDispatcher`: Solver selection
  - `DataValidator`: Input validation
- **Design**: Breaks down complex setup logic, improves testability

#### `jacobians.py` - Derivatives (39 lines)
- **Purpose**: Pseudo-Voigt derivative calculations
- **Functions**:
  - dG/dx, dG/dη, dL/dx, dL/dη
- **Design**: Minimal, used by pv_sum_with_jac

#### `mcmc_utils.py` - MCMC Diagnostics (38 lines)
- **Purpose**: ESS and R̂ calculations for Bayesian uncertainty
- **Functions**:
  - Effective Sample Size (ESS)
  - Potential Scale Reduction Factor (R̂)
- **Design**: Diagnostic helpers for MCMC trace analysis

#### `bounds_classic.py` - Classic Solver Bounds (58 lines)
- **Purpose**: Specialized bounds for curve_fit backend
- **Logic**: Minimal bounds with optional center windowing
- **Design**: Separate from modern bounds for simplicity

### 5.2 Fit Module (`fit/`)

#### `__init__.py` - Solver Dispatcher (134 lines)
- **Purpose**: Central dispatcher routing to appropriate solver
- **Function**: `solve(x, y, peaks, mode, baseline, options)`
- **Routing**:
  - Extract `options['solver']`
  - Delegate to classic/modern/modern_vp/lmfit
  - Handle missing optional backends gracefully
- **Design**: Unified interface for all solvers

#### `classic.py` - SciPy curve_fit Backend (328 lines)
- **Purpose**: Simple, unweighted least-squares solver
- **Algorithm**: Levenberg-Marquardt via `scipy.optimize.curve_fit`
- **Features**:
  - Lock-aware parameter packing
  - Minimal bounds (h≥0, w>eps, optional center clamp)
  - No robust loss or noise weighting
  - Step ▶ support via shared residuals
- **Design**: Simplicity and speed, good baseline solver

#### `modern.py` - Trust-Region-Reflective (292 lines)
- **Purpose**: Robust solver with flexible options
- **Algorithm**: `scipy.optimize.least_squares` with TRF method
- **Features**:
  - Robust losses: linear, soft_l1, huber, cauchy
  - Noise weighting: none, poisson, inv_y
  - Full bounds on all parameters
  - Parameter scaling for conditioning
  - Optional restart + jitter for multi-modal problems
  - Analytic Jacobian when available, FD fallback
- **Design**: Flexible, good for outliers and ill-conditioning

#### `modern_vp.py` - Variable-Projection (443 lines)
- **Purpose**: Specialized solver for ill-conditioned center/width problems
- **Algorithm**:
  - Heights solved exactly by NNLS: argmin_{h≥0} ||W(A h - y)||₂
  - Centers/widths updated via damped Gauss-Newton + backtracking
  - Interleaved optimization with step-halving
- **Features**:
  - Robust loss and weighting support (same as TRF)
  - Full lock support
  - Automatic non-negativity on heights
  - Backtracking and λ damping for stability
- **Design**: Robust for overlapping/overlaid peaks

#### `step_engine.py` - Single-Iteration Stepping (243 lines)
- **Purpose**: Implement "Step ▶" functionality
- **Algorithm**:
  1. Compute Jacobian at current θ
  2. Solve damped normal equations: (J'J + λI)δ = -J'r
  3. Evaluate cost at θ + δ
  4. Backtrack if cost not decreased
- **Output**: `StepResult(accepted, λ, backtracks, step_norm, reason)`
- **Design**: Shared across all solvers, same residuals/weights/bounds

#### `lmfit_backend.py` - Optional LMFIT Wrapper (233 lines)
- **Purpose**: Integration with optional lmfit library
- **Features**:
  - Wrap peaks into lmfit Parameters
  - Configure bounds and constraints
  - Use lmfit minimize with NLS method
  - Report NotAvailable if lmfit missing
- **Design**: Graceful fallback if lmfit absent

#### `base.py` - Solver Abstraction (277 lines)
- **Purpose**: Define common interface for all solvers (refactored)
- **Base Class**: `PeakFitSolver`
  - Abstract `solve()` method
  - Input validation
  - Result checking
  - Registry system for runtime discovery
- **Components**:
  - `SolveResult`: TypedDict for output structure
  - `SolverRegistry`: Runtime solver registration
  - `AdapterSolver`: Wrapper for migration
- **Design**: Enable new solvers without touching dispatcher

#### `bounds.py` - Bound Packing (157 lines)
- **Purpose**: Pack/unpack bounds respecting locks
- **Logic**:
  - Reduced bounds for locked parameters
  - Data-driven upper bounds (95th percentile for heights, 0.5×span for widths)
  - Optional center windowing
- **Design**: Supports lock-aware optimization

#### `result.py` - FitResult Container (26 lines)
- **Purpose**: Structured fit output
- **Fields**: θ, cost, RMSE, solver choice, iteration count, diagnostics
- **Design**: Minimal, clear results schema

#### `orchestrator.py` - Dispatch Logic (35 lines)
- **Purpose**: Route to appropriate solver
- **Design**: Thin wrapper around `__init__.solve()`

#### `utils.py` - Helper Utilities (31 lines)
- **Purpose**: MAD sigma, robust cost calculation
- **Functions**:
  - `mad_sigma()`: Median absolute deviation for outlier-robust estimate
  - `robust_cost()`: Cost with optional robust loss
- **Design**: Shared utilities for solver implementations

### 5.3 Batch Module (`batch/`)

#### `runner.py` - Batch Orchestrator (1,315 lines)
- **Purpose**: Process multiple files with consistent configuration
- **Workflow**:
  1. Glob files matching patterns (case-insensitive on Windows)
  2. For each file:
     - Load data
     - Compute baseline (ALS/poly, optional fit-window windowing)
     - Seed peaks (reuse current/template with optional re-height, or auto)
     - Fit using `core.fit_api.run_fit_consistent()`
     - Compute uncertainty
     - Write per-file outputs
  3. Combine peak tables into summary CSV
  4. Stream progress back to UI
- **Features**:
  - Peak seeding strategies: current, template (with auto-apply), auto
  - Optional per-file re-height from signal
  - Parallel uncertainty computation with worker capping
  - Per-file and batch summary CSV/TXT exports
  - Progress streaming and abort handling
- **Design**: Mirror single-file fit engine, ensure parity

### 5.4 UI Module (`ui/`)

#### `app.py` - Main Tkinter GUI (7,113 lines)
- **Purpose**: Interactive user interface
- **Layout**:
  - **Left pane**: Matplotlib canvas (plot area)
  - **Right pane**: Scrollable control panel
  - **Top-right**: Segmented action bar (File | Fit | Plot | Help)
  - **Bottom**: Status bar (progress + collapsible log)
- **Control Sections**:
  - **Baseline**: ALS/Polynomial method, λ/p/degree tuning
  - **Peaks Panel**: Add/remove/lock peaks, per-peak η, global broadcast
  - **Solver Selection**: Radio buttons for Classic/TRF/VP/LMFIT
  - **Solver Options**: Loss, weights, f_scale, maxfev, restarts, jitter
  - **Uncertainty**: Method (asymptotic/bootstrap/bayesian), n_samples, band
  - **Batch**: Patterns, source, output folder, progress
- **Key Features**:
  - Click-to-add peaks on plot (toggle mode)
  - Auto-seeding of prominent peaks
  - Real-time plot updating
  - Lock/unlock parameters per-peak
  - Per-peak and global η control
  - Uncertainty band overlay
  - Help window (F1) with scrollable content
- **Persistence**: Configuration saved to ui/ui_config/gl_peakfit_config.json
- **Design**: Monolithic but well-structured with clear sections

#### `helptext.py` - Help Documentation (~15k lines)
- **Purpose**: Comprehensive user documentation
- **Contents**: Mathematical background, quick-start, solver descriptions, export schemas

#### `uncertainty_utils.py` - UI Uncertainty Helpers (100 lines)
- **Purpose**: UI-specific uncertainty display functions
- **Functions**: Band smoothing, label formatting, visualization helpers

### 5.5 Infrastructure Module (`infra/`)

#### `performance.py` - Backend Switching (577 lines)
- **Purpose**: Defensive performance layer with graceful fallback
- **Backends**:
  - NumPy (default, always available)
  - Numba (optional JIT compilation, 2-5× speedup)
  - CuPy (optional GPU via CUDA, 10-100× speedup)
- **Features**:
  - Safe backend switching with try-except fallback
  - Shadow-compare mode for testing fast vs slow paths
  - Peak evaluation: `eval_total()`, `eval_components()`
  - Design matrix construction for NNLS
  - Thread pool and BLAS control via threadpoolctl
  - RNG seeding for reproducibility
- **Design**: Transparent acceleration, no breaking changes if GPU unavailable

#### `config.py` - Configuration Persistence (78 lines)
- **Purpose**: Load/save a config file passed as `path`; nothing calls it (the GUI's file is `ui/ui_config/gl_peakfit_config.json`, in `ui/app.py`)
- **Stored Settings**:
  - Baseline method and defaults (ALS/polynomial params)
  - Solver choice and options (loss, weights, f_scale, etc)
  - Uncertainty method and parameters (n_samples, band toggle)
  - UI toggles (legend, click-to-add, performance flags)
  - Global η, x-label
  - Batch defaults and templates
  - Performance flags (numba, gpu, cache, seed_all, max_workers)
- **Design**: Recursive merge for backward compatibility, key migrations

#### `logging.py` - Logging Setup (50 lines)
- **Purpose**: Configure logging for batch processing
- **Features**: Stream-based logging for UI integration

---

## 6. Data Flow & Workflows

### 6.1 Single-File Fitting Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Data (CSV/TXT/DAT)                                  │
│    ├─ Auto-detect delimiters                                │
│    ├─ Strip comments (#, %, //)                             │
│    ├─ Drop non-numeric columns                              │
│    ├─ Remove non-finite rows                                │
│    └─ Sort x ascending if needed                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 2. Define Fit Window (min/max x range)                      │
│    └─ Optional: baseline computed only on fit window        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 3. Compute Baseline (ALS or Polynomial)                     │
│    ├─ ALS: λ (smoothness), p (asymmetry), niter, tol       │
│    ├─ Poly: degree (auto-clamped), normalize_x              │
│    └─ If windowed: interpolate to full x                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 4. Add Peaks (manually or auto-seed)                        │
│    ├─ Click on plot → add peak at location (height auto)    │
│    ├─ Auto-seed: detect prominent local maxima             │
│    ├─ Lock center/width per-peak                            │
│    └─ Set per-peak or global η (Gaussian-Lorentzian mix)   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 5. Select Solver (Classic / TRF / VP / LMFIT)               │
│    └─ Configure options (loss, weights, bounds, etc)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 6. Fit (or Step ▶ for single iteration)                     │
│    ├─ Pack parameters respecting locks & bounds             │
│    ├─ Build residual r(θ) and optional Jacobian J(θ)        │
│    ├─ Apply noise weighting (none/poisson/inv_y)            │
│    ├─ Solve: minimize ||W·r(θ)||₂ or robust loss          │
│    └─ Unpack results θ* → {h, c, w, η} per peak            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 7. Compute Uncertainty (asymptotic/bootstrap/bayesian)      │
│    ├─ Asymptotic: Jacobian covariance + delta method        │
│    ├─ Bootstrap: resample residuals → refit → stats         │
│    ├─ Bayesian: MCMC posterior (emcee) + ESS/R̂ checks      │
│    └─ Optional prediction band (smoothed)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 8. Plot Results                                              │
│    ├─ Show raw data, baseline, fitted model, peaks          │
│    ├─ Overlay uncertainty band if selected                  │
│    └─ Display legend with peak centers                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 9. Export Results                                            │
│    ├─ _fit.csv: peak parameters + metadata                  │
│    ├─ _trace.csv: full x,y with baseline & components       │
│    ├─ _uncertainty.csv: param stats (means, SDs, CIs)        │
│    └─ _uncertainty.txt: human-readable with ± notation      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Batch Processing Workflow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Specify Configuration                                     │
│    ├─ Patterns: *.csv;*.txt;*.dat (semicolon-separated)     │
│    ├─ Peak source: current/template/auto                    │
│    ├─ Optional: re-height per file                          │
│    ├─ Save traces: per-file trace CSVs                      │
│    └─ Uncertainty method, solver, baseline config           │
└─────────────────┬──────────────────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────┐
│ 2. Glob File Matching                                      │
│    ├─ List directory recursively                           │
│    ├─ Match against patterns (case-insensitive on Windows)│
│    └─ Get file list to process                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────┐
│ 3. For Each File (parallel uncertain, outer loop)          │
│    │                                                       │
│    ├─ Load data (same cleaning as single)                  │
│    │                                                       │
│    ├─ Compute baseline                                    │
│    │  └─ Optional: compute only on fit window             │
│    │                                                       │
│    ├─ Seed peaks                                          │
│    │  ├─ Source=current: use peaks from GUI               │
│    │  │  └─ If reheight: refresh heights from signal      │
│    │  ├─ Source=template: use saved template              │
│    │  │  └─ Auto-apply if enabled                         │
│    │  └─ Source=auto: detect local maxima (up to max)     │
│    │                                                       │
│    ├─ Fit using run_fit_consistent()                      │
│    │  └─ (Same engine as GUI, ensures parity)             │
│    │                                                       │
│    ├─ Compute uncertainty                                 │
│    │  └─ Parallel bootstrap/Bayesian with worker cap      │
│    │                                                       │
│    └─ Write per-file outputs                              │
│       ├─ {filename}_fit.csv                               │
│       ├─ {filename}_trace.csv (if save_traces)            │
│       ├─ {filename}_uncertainty.csv                       │
│       └─ {filename}_uncertainty.txt                       │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────┐
│ 4. Combine Results                                          │
│    └─ Merge all peak tables → batch_summary.csv            │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Key Architectural Patterns

### 1. Lock-Aware Parameter Packing
**Pattern**: Parameters can be "locked" (held fixed) or "free" (optimized).
- Free parameters packed into reduced vector θ
- Locked parameters stored separately and held constant
- Benefits: Ensures locked params never drift; clear intent
- **Usage**: All solvers (Classic, TRF, VP), bootstrap resampling, jitter

### 2. Dual Residual/Jacobian Functions
**Pattern**: Build both r(θ) and (r(θ), J(θ)) to support analytic and finite-difference paths.
- Benefits: Solvers can choose; fallback to FD if analytic unavailable
- **Usage**: `core/residuals.py`, all solvers

### 3. Mode Isolation (Add vs Subtract)
**Pattern**: Single fitting engine handles both modes via inline residual logic.
- Add mode: r = (baseline + model) - y_raw
- Subtract mode: r = model - (y_raw - baseline)
- Benefits: No duplicate fitting logic
- **Usage**: All residual builders, step engine, exports

### 4. Solver Dispatch via Options Dict
**Pattern**: Options dict specifies solver; backends implement common interface.
- Benefits: Unified `solve(x, y, peaks, mode, baseline, options)` for all solvers
- **Usage**: `fit/__init__.py` dispatcher, batch runner

### 5. Uncertainty Result Container
**Pattern**: `UncertaintyResult` with canonical fields (method, label, stats, band, diagnostics).
- Benefits: Unified interface across Asymptotic/Bootstrap/Bayesian
- Backward compat: Dict-like `__getitem__` access
- **Usage**: UI display, export, batch summary

### 6. Defensive Backend Switching
**Pattern**: Try fast path (Numba/GPU) with immediate fallback to NumPy.
- Benefits: Graceful degradation; users don't notice if GPU unavailable
- **Usage**: `infra/performance.py`, peak evaluation

### 7. Shared Step Engine
**Pattern**: Single-iteration stepping uses same residuals/weights/bounds as full Fit.
- Benefits: Step ▶ visually "walks" the same landscape
- **Usage**: All solvers via `step_engine.py`

### 8. Deterministic Seeding
**Pattern**: RNG seed can be fixed globally or per-draw (seed_base + index).
- Benefits: Reproducible uncertainty estimates, batch parity
- **Usage**: Bootstrap, Bayesian, jitter, batch processing

### 9. Baseline Windowing
**Pattern**: Baseline optionally computed only on fit window, then interpolated to full x.
- Benefits: Improves local baseline fit, prevents rippling at edges
- Flag: `baseline_uses_fit_range` (default ON)
- **Usage**: Single fit, batch, all methods

### 10. CSV Schema Unification
**Pattern**: Single and batch export to same column order (no variations).
- Benefits: Downstream tooling needs one parser
- **Usage**: Single export, batch per-file export, batch summary

---

## 8. Solver Comparison & Selection

| Aspect | Classic | Modern TRF | Modern VP | LMFIT-VP |
|--------|---------|-----------|----------|----------|
| **Algorithm** | Levenberg-Marquardt | Trust-Region-Reflective | NNLS + Gauss-Newton | LM (lmfit) |
| **Robust Loss** | None | soft_l1, huber, cauchy | soft_l1, huber, cauchy | lmfit-supported |
| **Noise Weights** | None | poisson, inv_y | poisson, inv_y | None |
| **Bounds** | Minimal | Full | Full | Optional |
| **Heights** | Free | ≥0 with upper limit | NNLS (automatic ≥0) | Free |
| **Centers/Widths** | Free with optional window clamp | Full bounds | Full bounds | Free |
| **Lock Support** | Yes | Yes | Yes | Yes |
| **Speed** | Fast | Medium | Medium-slow | Depends |
| **Overlapping Peaks** | Struggles | Good | Excellent | Good |
| **Use Case** | Baseline testing | General purpose | Ill-conditioned | Alternative backend |
| **Step ▶ Support** | Yes | Yes | Yes | No (complex) |

---

## 9. Uncertainty Methods

### Asymptotic (Fast, Always Available)
- **Algorithm**: Jacobian-based covariance + delta method
- **Steps**:
  1. Compute Jacobian J at solution θ*
  2. Estimate σ² = RSS / dof
  3. Cov = σ² (J'J)⁻¹ (with Tikhonov regularization if needed)
  4. Form gradient G = ∂ŷ/∂θ on full x (FD)
  5. Band variance: Var[Ŷ] = diag(G Cov G')
  6. Smoothed 95% CI: ŷ ± 1.96 √Var
- **Advantages**: Fast, always available, no hyperparameters
- **Limitations**: Assumes linearity, may underestimate with nonlinearity

### Bootstrap (Distribution-Free, Captures Nonlinearity)
- **Algorithm**: Residual resampling → refit → parameter statistics
- **Steps**:
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
- **Limitations**: Computationally expensive

### Bayesian (MCMC, Optional emcee)
- **Algorithm**: MCMC posterior sampling with ESS/R-hat diagnostics
- **Steps**:
  1. Define log-likelihood: log π(θ|data) ∝ -0.5 RSS(θ)
  2. Define priors (flat with bounds)
  3. Run MCMC (emcee walkers) for burn-in + production
  4. Thin by autocorr; compute ESS, R̂ diagnostics
  5. Extract posterior mean/SD/95% credible intervals
  6. Optional: posterior predictive band
- **Diagnostics**: ESS (effective sample size), R̂ (potential scale reduction)
- **Advantages**: Posterior distributions, uncertainty on model assumptions
- **Limitations**: Requires emcee; MCMC tuning needed

---

## 10. Export Schemas

### A) Peak Table (`_fit.csv`)
```
file, peak, center, height, fwhm, eta, lock_width, lock_center,
area, area_pct, rmse, fit_ok, mode,
als_lam, als_p, fit_xmin, fit_xmax,
solver_choice, solver_loss, solver_weight, solver_fscale, solver_maxfev,
solver_restarts, solver_jitter_pct, step_lambda,
baseline_uses_fit_range, perf_numba, perf_gpu, perf_cache_baseline,
perf_seed_all, perf_max_workers
```

### B) Trace CSV (`_trace.csv`)
```
x, y_raw, baseline,
y_target_add, y_fit_add, peak1, peak2, ...,
y_target_sub, y_fit_sub, peak1_sub, peak2_sub, ...
```

### C) Uncertainty CSV (`_uncertainty.csv`)
```
param | mean | sd | p2_5 | p97_5 | ...
```

### D) Uncertainty TXT (`_uncertainty.txt`)
```
Method: Asymptotic
Label: Asymptotic 95% CI

Center [param]:     123.45 ± 0.12
Height [param]:     456.78 ± 2.34
...
```

---

## 11. Configuration Persistence

**File**: `ui/ui_config/gl_peakfit_config.json`, beside the code, so OneDrive syncs it; git-ignored. `GL_PEAKFIT_CONFIG` names another file. Until it exists, the old `~/.gl_peakfit_config.json` is read in its place and never changed.

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
  "solver_loss": "linear|soft_l1|huber|cauchy",
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

---

## 12. Testing Framework

**Test Suite**: 163 test files, comprehensive coverage across:

### Test Categories
1. **Solver tests**: Verify Classic/TRF/VP/LMFIT correctness and parity
2. **Baseline tests**: ALS, polynomial, fit-range behavior
3. **Uncertainty tests**: Asymptotic/bootstrap/Bayesian reproducibility
4. **Export tests**: CSV schema, no blank lines, header order
5. **Batch tests**: Single-batch parity, per-file outputs, progress streaming
6. **Performance tests**: Backend switching, Numba/GPU fallback, caching
7. **UI tests**: Config persistence, peak list updates, export dialogs
8. **Integration tests**: Full workflows (load → fit → export)

### Test Fixtures
- `two_peak_data`: Synthetic 2-peak spectrum
- `real1.csv`, `real2.csv`: Real spectral data
- `batch_small/`: Sample files for batch testing

### Testing Utilities
- `conftest.py`: Pytest configuration, RNG fixture
- `tools/smoke_uncertainty.py`: Manual uncertainty verification
- `tools/smoke_batch.py`: Manual batch processing test

---

## 13. Performance Optimization

### Backend Switching
- **NumPy**: Default, always available
- **Numba**: Optional JIT compilation (2-5× speedup on peak eval)
- **CuPy**: Optional GPU via CUDA (10-100× speedup if available)

### Optimization Strategies
- **NNLS for heights**: Avoids ill-conditioning of full parameter space
- **Variable projection**: Reduces effective dimensionality
- **Parameter scaling**: Improves numerical conditioning in TRF
- **Sparse baseline operators**: O(n) tri-diagonal solve for ALS
- **Baseline caching**: Optional cache of polynomial baseline
- **BLAS limiting**: Cap threads per BLAS operation when running parallel workers

### Threading & Multiprocessing
- **Batch**: ProcessPoolExecutor for uncertainty (bootstrap/Bayesian draws)
- **Worker cap**: Configurable `perf_max_workers` (default: CPU count - 1)
- **Strategy**: "outer" (parallelize batch files), "inner" (parallelize bootstrap draws)
- **BLAS control**: `threadpool_limits` from `threadpoolctl`

---

## 14. Recent Refactoring Improvements

### Completed Refactorings
1. **Parameter Utilities** (`core/param_utils.py`): Centralized packing/unpacking logic
   - 60% reduction in duplication
   - Single source of truth for parameter transformations

2. **Array Utilities** (`core/array_utils.py`): Safe, validated array operations
   - Reduced `.reshape(-1)` patterns by 35+ locations
   - Consistent error messages

3. **Typed Fitting Context** (`core/fit_context.py`): Replace dict with dataclass
   - 22 typed attributes with defaults
   - Dictionary-like access for backward compatibility
   - Reduces defensive null checks

4. **Solver Abstraction Base** (`fit/base.py`): Abstract base class + adapter pattern
   - Consistent solver interface
   - Runtime solver registration/discovery
   - Clear migration path

5. **Fitting Workflow Orchestration** (`core/fit_orchestrator.py`): Modular components
   - BaselineConfig, BaselineCompute, PeakInitializer, SolverDispatcher, DataValidator
   - Complexity reduced by 70+ branches
   - Each component independently testable

### Impact
- **Duplication**: 40% → ~15% (62% reduction)
- **Type hint coverage**: 25% → 50%+ (100% expansion)
- **Testability**: Poor → Excellent (modular)
- **Code reuse**: Low → High (centralized)
- **Backward compatibility**: 100% maintained (all changes additive)

---

## 15. Known Limitations

### Mathematical
1. Pseudo-Voigt only (no other peak shapes)
2. Single baseline across all peaks
3. Fixed η (not optimized, user-set)
4. Linear fit window only

### Solver
1. Classic: No robust loss
2. VP: Requires non-negative heights
3. Bounds stiffness near limits
4. Restart jitter may not escape all local minima

### Uncertainty
1. Asymptotic: Assumes linearity
2. Bootstrap: Assumes exchangeable residuals
3. Bayesian: Requires emcee, needs tuning
4. Band smoothing: Can mask fine structure

### GUI
1. Large datasets (100k+ points) slow plotting
2. No interactive peak dragging
3. No multi-file side-by-side view
4. No undo/redo history

### Batch
1. Large files may exhaust memory
2. Pattern matching case-sensitive on Unix
3. Partial failure doesn't skip remaining files

---

## 16. How to Extend the Codebase

### Adding a New Solver
1. Create `fit/my_solver.py` implementing `solve(x, y, peaks, mode, baseline, options)`
2. Import and register in `fit/__init__.py`
3. Optionally inherit from `PeakFitSolver` base class (in `fit/base.py`)
4. Test against `two_peak_data` fixture
5. Add to UI radio buttons

### Adding Uncertainty Method
1. Create `uncertainty/my_method.py` with method implementation
2. Create `UncertaintyResult` with stats/bands/diagnostics
3. Register in `core/uncertainty_router.py`
4. Add to UI uncertainty method selection

### Adding Baseline Method
1. Implement function in `core/signals.py`
2. Update `fit_orchestrator.py` to handle new method
3. Add configuration options to `infra/config.py`
4. Add to UI baseline method selection

### Customizing UI
1. Modify `ui/app.py` for layout/controls
2. Update `ui/helptext.py` for documentation
3. Adjust `infra/config.py` for new persistent options

---

## Summary

The **Interactive Peak Fit GUI** is a comprehensive, well-architected scientific software project demonstrating:

1. **Modular Design**: Clear separation between core fitting, UI, batch, and infrastructure
2. **Multiple Solver Strategies**: From simple (curve_fit) to sophisticated (Variable-Projection with NNLS)
3. **Sophisticated Uncertainty**: Three complementary methods with comprehensive diagnostics
4. **Reproducibility**: Deterministic seeding, lock/bound enforcement, extensive tests
5. **Performance Flexibility**: Optional GPU/Numba with safe fallbacks
6. **User-Friendly**: Interactive GUI with real-time feedback, batch processing, detailed exports
7. **Robustness**: Defensive programming, 163 test files, comprehensive error handling
8. **Maintainability**: Recent refactoring (5 new modules, 1,310 lines) improving type safety and reducing duplication by 62%

The codebase (~29,614 lines across 214+ files) balances feature richness with maintainability through clear abstractions, comprehensive testing, and thoughtful architectural patterns.
