# Graph Report - Interactive Peak Fit GUI  (2026-08-18)

## Corpus Check
- 233 files · ~125,286 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1893 nodes · 4203 edges · 130 communities (121 shown, 9 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 149 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2c91fecc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- PeakFitApp
- ensure_unc_common
- performance.py
- PeakFitApp
- test_theme_application.py
- .refresh_plot
- ._compute_uncertainty_sync
- BaselineCompute
- 5.1 Core Module (`core/`)
- bootstrap_ci
- PeakFitSolver
- runner.py
- ._build_ui
- Any
- test_z_audit.py
- route_uncertainty
- pseudo_voigt
- uncertainty.py
- bayesian_ci
- app.py
- main
- pv_sum
- data_io.py
- step_engine.py
- test_step_parity_trf_vp_lmfit.py
- peaks.py
- Peak
- pack_theta_bounds
- fit/__init__.py
- signals.py
- array_utils.py
- write_uncertainty_txt
- UncertaintyResult
- bootstrap.py
- predict_full
- lmfit_backend.py
- FitContext
- residuals.py
- build_residual
- test_bayes_mcse_export.py
- modern_vp.py
- param_utils.py
- Interactive Peak Fit GUI (Gaussian–Lorentzian / pseudo-Voigt)
- modern.py
- ScrollableFrame
- asymptotic_ci
- run_diagnostics
- test_perf_seed_drives_uncertainty.py
- method_display_to_key
- test_bayes_diagnostics_toggle.py
- test_labels_inline.py
- ._apply_performance_now
- build_trace_table
- verify_unc_cleanup.py
- test_batch_baseline_method_column.py
- test_batch_bootstrap_band_toggle.py
- TestConfigureNamedFonts
- TestFontCandidates
- test_labels.py
- test_log_buffer.py
- test_cli_baseline_method_exports_metadata
- test_bayes_seed_determinism.py
- Theme Implementation Summary
- TestAppStartupWithTheme
- write_batch_uncertainty_long
- Improved Plan: Tkinter Font Rendering / Antialiasing
- test_shadow_compare.py
- test_unc_normalize_result.py
- save_config
- set_gpu
- test_config_migration_baseline_defaults.py
- Code Refactoring Summary
- test_step_no_freeze.py
- test_unc_label_aliases.py
- test_unc_stats_keys_consistent.py
- helptext.py
- make_bounds_classic
- Interactive Peak Fit GUI - Complete Codebase Documentation
- 5. Key Modules
- ui/__init__.py
- test_smoke_gui_vs_batch_unc.py
- run_batch
- Performance Apply Debounce Pattern - Knowledge Base
- 15. Important Design Patterns
- 7. Key Architectural Patterns
- Interactive Peak Fit GUI - Comprehensive Codebase Analysis
- Issue: Missing Import Error in Type Annotations
- 7. Fitting System
- design_matrix
- .current_fit_mask
- test_progress_logging.py
- CLAUDE Memory — Interactive Peak Fit GUI
- Export Formats
- blas_limit_ctx
- test_poly_baseline_caching.py
- test_unc_bootstrap_outputs.py
- test_batch_runner.py
- ._update_bootstrap_visibility
- 11. Performance/Optimization
- 16. Known Limitations
- Three Methods
- 15. Known Limitations
- TestThemeWithDefaultTkinter
- 14. Dependencies
- 3. Core Functionality
- 9. Batch Processing
- 10. Export Schemas
- 16. How to Extend the Codebase
- 3. Core Technologies & Dependencies
- test_bootstrap_workers.py
- test_template_auto_apply.py
- test_solver_parity.py
- test_solver_regression.py
- 12. Testing
- 12. Testing Framework
- 13. Performance Optimization
- 9. Uncertainty Methods
- build_peak_table
- TestRequirementsUpdated
- TestThemeImport
- 14. Recent Refactoring Improvements
- 1. Project Purpose & Overview
- 4. Mathematical Foundation
- Changelog
- test_band_without_model.py
- test_bootstrap_vector_aliases.py
- test_curve_band_alias.py

## God Nodes (most connected - your core abstractions)
1. `PeakFitApp` - 202 edges
2. `Peak` - 156 edges
3. `bootstrap_ci()` - 68 edges
4. `pv_sum()` - 65 edges
5. `run_batch()` - 59 edges
6. `PeakFitApp` - 58 edges
7. `bayesian_ci()` - 49 edges
8. `pseudo_voigt()` - 47 edges
9. `run_fit_consistent()` - 42 edges
10. `pack_theta_bounds()` - 34 edges

## Surprising Connections (you probably didn't know these)
- `test_fit_api_no_solver_fallbacks()` --uses--> `Peak`  [INFERRED]
  tests/test_unc_cleanup_smoke.py → core/peaks.py
- `_make_predict_full_generic()` --indirect_call--> `predict_full()`  [INFERRED]
  core/fit_api.py → batch/runner.py
- `bootstrap_ci()` --calls--> `predict_full()`  [INFERRED]
  core/uncertainty.py → batch/runner.py
- `test_bayes_ci_determinism_seed_all_true()` --indirect_call--> `predict_full()`  [INFERRED]
  tests/test_bayes_ci_determinism.py → batch/runner.py
- `test_bayesian_vector_stats_have_underscore_aliases()` --indirect_call--> `predict_full()`  [INFERRED]
  tests/test_bayes_vector_aliases.py → batch/runner.py

## Import Cycles
- 3-file cycle: `core/__init__.py -> core/uncertainty.py -> core/fit_api.py -> core/__init__.py`
- 3-file cycle: `core/data_io.py -> core/uncertainty.py -> core/fit_api.py -> core/data_io.py`
- 4-file cycle: `core/__init__.py -> core/data_io.py -> core/uncertainty.py -> core/fit_api.py -> core/__init__.py`

## Communities (130 total, 9 thin omitted)

### Community 0 - "PeakFitApp"
Cohesion: 0.06
Nodes (20): als_baseline(), gaussian(), load_config(), load_xy_any(), lorentzian(), main(), pack_params(), Peak (+12 more)

### Community 1 - "ensure_unc_common"
Cohesion: 0.18
Nodes (20): bayes_knobs(), bootstrap_cfg(), ensure_unc_common(), Canonical knobs for strict router: - walkers: 0 means 'auto', Fill uncertainty flags used by GUI/Batch parity and bands/diagnostics. Keeps…, test_batch_band_written(), test_batch_bootstrap_runs_with_arrays(), _fake_fit_ctx() (+12 more)

### Community 2 - "performance.py"
Cohesion: 0.12
Nodes (22): _eval_components_backend(), _eval_components_gpu(), _eval_components_numba(), _eval_components_numpy(), _eval_total_backend(), _eval_total_gpu(), _eval_total_numba(), _eval_total_numpy() (+14 more)

### Community 3 - "PeakFitApp"
Cohesion: 0.05
Nodes (12): test_no_theme_hooks(), test_classic_advanced_collapsed(), test_unc_workers_vs_seed_all(), test_asymptotic_no_runtime_warning(), PeakFitApp, Mirror single-file BLAS/threads policy during batch uncertainty., Disable click-to-add regardless of checkbox state., Restore click-to-add to match the current checkbox state. (+4 more)

### Community 4 - "test_theme_application.py"
Cohesion: 0.08
Nodes (21): Test suite for theme application functionality. Tests ensure that: 1. Theme…, Test that theme loading is non-fatal to app startup., Test that apply_theme never raises exceptions., Test that logging theme status never crashes., Test theme integration with the main app., Test that PeakFitApp can be instantiated after theme is applied., Test theme application with missing dependencies., Test that apply_theme returns a tuple of (bool, str). (+13 more)

### Community 5 - ".refresh_plot"
Cohesion: 0.15
Nodes (3): Persist FWHM cap immediately. If 'All' is on, apply to every peak., Enter key in the cap entry: if 'All' is on, propagate to all; else edit…, Persist current cap UI prefs and options to the config file.

### Community 6 - "._compute_uncertainty_sync"
Cohesion: 0.10
Nodes (10): log_action(), Path, Decorator to emit debug log lines when debug mode is on., Return a stable canonical label for an uncertainty method key., Return the canonical uncertainty method key., Compute uncertainty synchronously for batch processing., Write the current uncertainty band to a CSV file., Process one spectrum: reset band, fit, export fit/trace, then compute and… (+2 more)

### Community 7 - "BaselineCompute"
Cohesion: 0.06
Nodes (25): BaselineCompute, BaselineConfig, DataValidator, PeakInitializer, Any, ndarray, Peak, Compute ALS baseline with optional windowing. (+17 more)

### Community 8 - "5.1 Core Module (`core/`)"
Cohesion: 0.05
Nodes (41): 5.1 Core Module (`core/`), 5.2 Fit Module (`fit/`), 5.3 Batch Module (`batch/`), 5.4 UI Module (`ui/`), 5.5 Infrastructure Module (`infra/`), 5. Main Components & Responsibilities, `app.py` - Main Tkinter GUI (7,113 lines), `array_utils.py` - Array Operations (273 lines) (+33 more)

### Community 9 - "bootstrap_ci"
Cohesion: 0.08
Nodes (22): bootstrap_ci(), Residual bootstrap with robust refitting and clear diagnostics. Notes ----- *…, test_batch_bootstrap_produces_band_when_requested(), test_band_cpu_parallel_matches_sequential(), test_bootstrap_batch_refit_requires_ok_flag(), test_batch_refit_errors_exposed(), test_diag_reports_n_boot(), test_gui_like_single_file_jitter_is_applied() (+14 more)

### Community 10 - "PeakFitSolver"
Cohesion: 0.08
Nodes (24): ABC, AdapterSolver, PeakFitSolver, Any, callable, ndarray, Peak, TypedDict (+16 more)

### Community 11 - "runner.py"
Cohesion: 0.10
Nodes (20): Batch processing helpers for Peakfit 3.x., Batch processing orchestrator for Peakfit 3.x. This module implements a…, Compatibility wrapper using legacy signature., # IMPORTANT: align everything to the fit grid so Y, residuals and band share…, run(), pv_area(), Spectral peak models for Peakfit 3.x. Only pseudo-Voigt support is sketched out…, Return the analytic area of a pseudo-Voigt peak. (+12 more)

### Community 12 - "._build_ui"
Cohesion: 0.09
Nodes (13): add_tooltip(), Add a 'Compute band' checkbox just under the 'Show band' checkbox., Attach a simple tooltip to ``widget`` displaying ``text``., Clamp bootstrap jitter to a safe [0..0.25] range., Extract Bootstrap + perf + band prefs mirroring GUI settings., normalize_float(), normalize_int(), Any (+5 more)

### Community 13 - "Any"
Cohesion: 0.14
Nodes (27): _as_mapping(), build_export_artifacts(), _canonical_unc_label(), _ensure_result(), _extract_stats_table(), _format_unc_text(), _iter_param_rows(), _iter_peak_rows_wide() (+19 more)

### Community 14 - "test_z_audit.py"
Cohesion: 0.83
Nodes (3): _hash(), _parity_ratio(), test_audit_line()

### Community 15 - "route_uncertainty"
Cohesion: 0.10
Nodes (27): _infer_alpha(), _norm_jitter(), _normalize_model_eval(), Any, ndarray, Route to the appropriate uncertainty engine and return an UncertaintyResult-…, Pull a CI alpha from a fit context if present; default 0.05 (95%)., Accept jitter as either a fraction (0..1) or percent (0..100). If > 1.5, treat… (+19 more)

### Community 16 - "pseudo_voigt"
Cohesion: 0.14
Nodes (18): test_export_fit_columns(), _sync_run(), test_export_uncertainty_single(), skipif, test_legend_toggle_smoke(), test_single_export_always_computes_uncertainty(), test_single_export_band_written(), test_single_export_ignores_history_and_uses_ui_method() (+10 more)

### Community 17 - "uncertainty.py"
Cohesion: 0.08
Nodes (26): ess_autocorr(), ndarray, rhat_split(), _build_residual_vector(), _coerce_draws_to_thetas(), complex_step_jacobian(), _norm_solver_and_sharing(), _param_df() (+18 more)

### Community 18 - "bayesian_ci"
Cohesion: 0.10
Nodes (23): bayesian_ci(), Sample posterior of free θ and σ with emcee. Deterministic with seed. Returns…, _dummy_predict(), skipif, test_bayes_band_gating(), skipif, test_bayes_band_subset_deterministic(), _fixture() (+15 more)

### Community 19 - "app.py"
Cohesion: 0.08
Nodes (23): Convenience launcher for the Peakfit 3.x prototype UI. Ensures that package…, test_baseline_method_persistence(), test_config_persistence_perf_unc(), Verify default config values and persistence of toggles., test_defaults_persist(), _make_app(), test_span_cancel_on_leave_restores_state(), test_span_preserves_toggle_off() (+15 more)

### Community 20 - "main"
Cohesion: 0.08
Nodes (25): Tests for font configuration module., Test pick_first_available_family function., When no candidate fonts exist, return None., Test configure_ttk_styles function., Should handle None sans_family gracefully., Should configure Fit.TButton with bold variant., Test font caching behavior., Results should be cached per root window id. (+17 more)

### Community 21 - "pv_sum"
Cohesion: 0.14
Nodes (26): pv_sum(), Return the sum of pseudo-Voigt peaks evaluated at ``x``. ``peaks`` should be an…, _build_residual(), iterate(), _pack(), prepare_state(), ndarray, Peak (+18 more)

### Community 22 - "data_io.py"
Cohesion: 0.09
Nodes (23): _as_scalar_or_none(), _finite(), _get_pct(), iter_uncertainty_rows(), _normalize_param_stats(), peaks_to_dicts(), Data I/O helpers for Peakfit 3.x. This module provides functions to load…, Build long-format rows with guaranteed canonical labels and p2_5/p97_5.… (+15 more)

### Community 23 - "step_engine.py"
Cohesion: 0.12
Nodes (23): pv_and_grads(), Evaluate a pseudo-Voigt peak and its gradients. Parameters ---------- x :…, Backward compatibility wrappers for weight helpers., combine_weights(), Return IRLS weights for ``resid`` under ``loss``. ``loss`` follows the…, Combine noise (``wn``) and robust (``wr``) weights. ``None`` inputs are treated…, robust_weights(), iterate() (+15 more)

### Community 24 - "test_step_parity_trf_vp_lmfit.py"
Cohesion: 0.33
Nodes (12): lmfit_step(), modern_trf_step(), modern_vp_step(), Peak, StepResult, _vp_initial_cost(), noise_weights(), r"""Return element-wise noise weights for ``y``. ``mode`` may be ``"none"``… (+4 more)

### Community 25 - "peaks.py"
Cohesion: 0.08
Nodes (28): build_residual_and_jacobian(), classic_step(), _loss_weights(), _make_predict_full_generic(), pseudo_voigt(), ndarray, Return square-root robust loss weights for residual ``r``. Parameters…, Construct residual/Jacobian consistent with the solver settings. Parameters… (+20 more)

### Community 26 - "Peak"
Cohesion: 0.17
Nodes (12): Peak, Simple container describing a pseudo-Voigt peak., prepare_state(), test_batch_uncertainty_summary(), test_modern_vp_iterate_respects_bounds_and_reference(), synthetic_data(), test_fallback_solvers_smoke(), _load() (+4 more)

### Community 27 - "pack_theta_bounds"
Cohesion: 0.12
Nodes (14): make_bounds(), make_bounds_classic(), ndarray, Return bounds and clipped start vector for the Classic solver., _trf_prepare(), pack_theta_bounds(), PeakLike, ndarray (+6 more)

### Community 28 - "fit/__init__.py"
Cohesion: 0.16
Nodes (18): _adapt_result(), iterate(), _peaks_from_theta(), ndarray, Unified solver interface for Peakfit 3.x., Dispatch to the selected backend and return a :class:`FitResult`. The…, Dispatch to the backend iterate function., Return deep-copied peaks updated from a flattened ``theta`` vector. (+10 more)

### Community 29 - "signals.py"
Cohesion: 0.11
Nodes (19): Orchestration logic for fitting workflow. This module encapsulates the…, Core utilities for Peakfit 3.x. This package exposes the main core modules so…, als_baseline(), polynomial_baseline(), ndarray, Signal processing helpers for Peakfit 3.x. Currently provides stubs for…, Estimate an asymmetric least squares (ALS) baseline for ``y``. Parameters…, Return a robust signal-to-noise estimate for ``y``. Noise is estimated from the… (+11 more)

### Community 30 - "array_utils.py"
Cohesion: 0.15
Nodes (18): atleast_2d_column(), broadcast_to_shape(), ensure_1d_array(), ensure_vector(), extract_diagonal_plus(), match_shapes(), ndarray, Utilities for consistent array handling and shape validation. This module… (+10 more)

### Community 31 - "write_uncertainty_txt"
Cohesion: 0.12
Nodes (17): derive_export_paths(), export_uncertainty_pair(), Path, Write rows with no blank lines and stable header ordering., Write the human-readable TXT., Write batch-wide uncertainty CSV with appended MCSE columns., Public wrapper for writing a human-readable uncertainty report. Delegate to…, Return canonical export file paths based on ``user_path``. ``user_path`` may… (+9 more)

### Community 32 - "UncertaintyResult"
Cohesion: 0.24
Nodes (5): Any, Light weight container for uncertainty estimates. The class exposes the…, _require_nonneg_int(), _require_positive_int(), UncertaintyResult

### Community 33 - "bootstrap.py"
Cohesion: 0.16
Nodes (16): apply_global_seed(), Apply deterministic seeding across supported RNGs., _residual_builder_factory(), test_bootstrap_determinism_seed_all_true(), parametrize, _residual_builder_factory(), test_blas_env_clamp_pass_through(), bootstrap() (+8 more)

### Community 34 - "predict_full"
Cohesion: 0.12
Nodes (14): _auto_seed(), predict_full(), ndarray, Peak, Evaluate the full model for ``theta`` on ``x``., Return up to ``max_peaks`` automatically seeded peaks., pseudo_voigt(), Canonical pseudo-Voigt: eta*L + (1-eta)*G, parameterized by FWHM. - x: array-… (+6 more)

### Community 35 - "lmfit_backend.py"
Cohesion: 0.19
Nodes (13): iterate(), prepare_state(), ndarray, Peak, TypedDict, LMFIT backend using variable projection., Perform a single iteration using the lmfit backend., solve() (+5 more)

### Community 36 - "FitContext"
Cohesion: 0.15
Nodes (10): FitContext, Any, Typed context container for fitting operations. This module provides a…, Dictionary-like access for backward compatibility. Parameters ---------- key :…, Dictionary-like assignment for backward compatibility. Parameters ----------…, Get value with default fallback. Parameters ---------- key : str Attribute…, Convert to dictionary representation. Returns ------- dict[str, Any] All…, Typed context for fitting operations. This dataclass consolidates all fitting… (+2 more)

### Community 37 - "residuals.py"
Cohesion: 0.17
Nodes (14): build_residual_jac(), finite_diff_jacobian(), jacobian_fd(), ndarray, Residual and Jacobian construction for Peakfit 3.x., Finite-difference Jacobian for ``residual_fn``. Works with either NumPy or CuPy…, Forward-difference Jacobian of ``residual_fn`` evaluated at ``theta``., Return residual and Jacobian builder for solvers. The parameter vector… (+6 more)

### Community 38 - "build_residual"
Cohesion: 0.17
Nodes (11): build_residual(), Peak, Return a residual function ``r(theta)`` for the given data. ``theta`` is…, test_bayes_determinism_seed_all_true(), test_bayes_blas_env_clamp_pass_through(), bayesian(), TypedDict, Bayesian uncertainty estimation via MCMC sampling. (+3 more)

### Community 39 - "test_bayes_mcse_export.py"
Cohesion: 0.29
Nodes (13): _mcse_quantile(), Batch-replicate MCSE for a quantile (q in [0,1]). Returns NaN if insufficient…, _batch_rows_from_result(), _fake_asymp_result(), _fake_bayes_result(), parametrize, Path, _read_header() (+5 more)

### Community 40 - "modern_vp.py"
Cohesion: 0.19
Nodes (14): pv_design_matrix(), pv_sum_with_jac(), ndarray, Return design matrix for height-only fits. Each column ``k`` is the pseudo-…, Return ``y`` and analytic Jacobian for pseudo-Voigt peaks. Parameters are…, create_parameter_unpacker(), callable, Create a function to unpack (c, w) from a reduced parameter vector. This… (+6 more)

### Community 41 - "param_utils.py"
Cohesion: 0.19
Nodes (12): pack_nonlinear_params(), ndarray, Peak, Utilities for parameter packing/unpacking and structural handling. This module…, Pack (c, w) values into a reduced parameter vector. Inverse operation of the…, Validate that theta vector matches peak count and parameter structure.…, Vectors for nonlinear (c, w) parameters in variable-projection solver.…, Unpack center and width parameters from nonlinear vector. This function… (+4 more)

### Community 42 - "Interactive Peak Fit GUI (Gaussian–Lorentzian / pseudo-Voigt)"
Cohesion: 0.07
Nodes (26): A) Peak Table `_fit.csv` (single & batch use the same columns and order), B) Trace `_trace.csv` (per spectrum; single & batch), Baseline methods, Batch Processing, C) Uncertainty, Choosing a method (GUI), Choosing a method (programmatic / batch), Citation (+18 more)

### Community 43 - "modern.py"
Cohesion: 0.23
Nodes (12): ndarray, TypedDict, Modern solver with robust losses and analytic Jacobian., solve(), SolveResult, _to_solver_vectors(), mad_sigma(), ndarray (+4 more)

### Community 45 - "asymptotic_ci"
Cohesion: 0.14
Nodes (14): DataFrame, Write ``df`` to ``path`` without introducing extra blank lines., write_dataframe(), asymptotic_ci(), Return asymptotic parameter statistics and a prediction band. ``residual`` and…, _Optional, no_blank_lines(), fixture (+6 more)

### Community 46 - "run_diagnostics"
Cohesion: 0.19
Nodes (12): Frame, _add_font_sample(), _log(), Any, Tk, Font diagnostics module for Interactive Peak Fit GUI. Provides diagnostic…, Open a Toplevel window showing sample text in various font families. Args:…, Helper to add a font sample label with resolved family info. (+4 more)

### Community 47 - "test_perf_seed_drives_uncertainty.py"
Cohesion: 0.27
Nodes (7): _DummySamplerBand, _fake_predict(), _fake_resid(), test_bayesian_band_gated_without_force(), test_bayesian_band_produces_subset_when_forced(), test_bayesian_respects_seed(), test_bootstrap_uses_global_seed()

### Community 48 - "method_display_to_key"
Cohesion: 0.36
Nodes (7): test_method_display_to_key_asymptotic_default(), test_method_display_to_key_bayesian_prefix(), test_method_display_to_key_bootstrap_label(), test_method_display_to_key_passthrough_for_unknown(), method_display_to_key(), Utility helpers for uncertainty configuration mapping and validation., Map a GUI display label to a canonical configuration key. Parameters ----------…

### Community 49 - "test_bayes_diagnostics_toggle.py"
Cohesion: 0.33
Nodes (5): _basic_args(), DummySampler, _setup_sampler(), test_bayes_diag_toggle_off(), test_bayes_diag_toggle_on()

### Community 50 - "test_labels_inline.py"
Cohesion: 0.42
Nodes (8): f(), test_escapes(), test_existing_math_preserved(), test_mixed(), test_plain_text_passthrough(), test_subscripts(), test_superscripts(), test_toggle_off()

### Community 51 - "._apply_performance_now"
Cohesion: 0.11
Nodes (11): Return effective worker count for fitting., Return effective worker count for uncertainty pipelines., UI helper: actual clamping is handled in the performance layer., Request a performance apply with debounce for trace-driven calls. Args: source:…, Compute a signature for current performance state (for log dedupe)., Public wrapper for backward compatibility., Internal core apply with logging dedupe logic., Helper for Apply button - always runs immediately with logging. (+3 more)

### Community 52 - "build_trace_table"
Cohesion: 0.29
Nodes (7): build_trace_table(), load_xy(), ndarray, Load two-column numeric data from ``path``. The function accepts ``.txt``,…, Return a CSV trace table matching the v2.7 schema. Columns (in order): ``x,…, _headers(), test_trace_schema_v27()

### Community 53 - "verify_unc_cleanup.py"
Cohesion: 0.43
Nodes (7): check_batch(), check_core(), check_fit_api(), _contains(), main(), Path, read()

### Community 54 - "test_batch_baseline_method_column.py"
Cohesion: 0.57
Nodes (7): _one_peak(), Path, _read_first_row(), _simple_signal(), test_batch_csv_has_baseline_method_and_fields_for_als(), test_batch_csv_has_baseline_method_and_fields_for_polynomial(), _write_xy()

### Community 55 - "test_batch_bootstrap_band_toggle.py"
Cohesion: 0.50
Nodes (7): _base_config(), _install_bootstrap_stub(), fixture, Path, sample_csv(), test_bootstrap_band_file_respects_toggle_off(), test_bootstrap_band_file_respects_toggle_on()

### Community 56 - "TestConfigureNamedFonts"
Cohesion: 0.25
Nodes (5): Should configure font family when provided., Should not set family when None is provided., Test configure_named_fonts function., Should handle missing named fonts gracefully., TestConfigureNamedFonts

### Community 57 - "TestFontCandidates"
Cohesion: 0.25
Nodes (5): Test that font candidate lists are properly defined., Sans-serif candidates list should not be empty., Monospace candidates list should not be empty., DEFAULT_UI_SIZE should be an integer., TestFontCandidates

### Community 58 - "test_labels.py"
Cohesion: 0.46
Nodes (7): f(), test_escaped_literals_not_wrapped(), test_existing_math_passthrough(), test_multiple_fragments_mixed(), test_plain_text_passthrough(), test_subscript_fragment(), test_superscript_fragment_braced_spaces_trimmed()

### Community 59 - "test_log_buffer.py"
Cohesion: 0.52
Nodes (5): requires_display, make_app(), test_format_asymptotic_summary(), test_log_buffer_and_console(), test_log_threadsafe()

### Community 60 - "test_cli_baseline_method_exports_metadata"
Cohesion: 0.48
Nodes (6): _is_nan_str(), parametrize, Path, _read_fit_rows(), test_cli_baseline_method_exports_metadata(), _write_synthetic()

### Community 61 - "test_bayes_seed_determinism.py"
Cohesion: 0.67
Nodes (5): _extract_stat_vector(), _fixture(), skipif, test_bayesian_different_seeds_change_output_somewhat(), test_bayesian_seed_is_deterministic()

### Community 62 - "Theme Implementation Summary"
Cohesion: 0.08
Nodes (23): 1. **Created: `ui/theme.py`**, 2. **Updated: `requirements.txt`**, 3. **Integrated: `ui/app.py`**, After (With Themes Installed):, apply_theme() Function Behavior, Backward Compatibility, Before (Default):, Changes Made (+15 more)

### Community 63 - "TestAppStartupWithTheme"
Cohesion: 0.33
Nodes (4): Test that app starts successfully with theme integration., Verify main() function exists in ui.app., Verify main imports apply_theme., TestAppStartupWithTheme

### Community 64 - "write_batch_uncertainty_long"
Cohesion: 0.50
Nodes (4): Write aggregated rows to: - batch_uncertainty_long.csv (preferred) -…, write_batch_uncertainty_long(), _all_finite(), test_batch_summary_no_nan()

### Community 65 - "Improved Plan: Tkinter Font Rendering / Antialiasing"
Cohesion: 0.09
Nodes (21): Codebase Context (What Was Found), Config keys to add to `DEFAULTS` dict (ui/app.py line 408), `configure_named_fonts`, `configure_ttk_styles`, Hook into startup, Implementation Order, Improved Plan: Tkinter Font Rendering / Antialiasing, Integration into `ui/app.py` (+13 more)

### Community 66 - "test_shadow_compare.py"
Cohesion: 0.60
Nodes (4): skipif, _reload_performance(), test_shadow_compare_ok(), test_shadow_compare_warns_and_falls_back()

### Community 68 - "save_config"
Cohesion: 0.11
Nodes (4): format_axis_label_inline(), Store per-method Show/Compute preferences in the config., r"""Convert ``^``/``_`` fragments to inline math while preserving normal text.…, save_config()

### Community 69 - "set_gpu"
Cohesion: 0.21
Nodes (19): Enable Numba JIT backend when available., Enable CuPy GPU backend when available., Return the current numeric backend label (e.g., 'numpy', 'cupy', 'numba')., set_gpu(), set_numba(), _update_backend(), which_backend(), skipif (+11 more)

### Community 70 - "test_config_migration_baseline_defaults.py"
Cohesion: 0.83
Nodes (3): _import_app(), test_legacy_keys_are_migrated_to_baseline_defaults(), test_roundtrip_persists_method_and_per_method_defaults()

### Community 71 - "Code Refactoring Summary"
Cohesion: 0.10
Nodes (19): 1. Parameter Handling Utilities (`core/param_utils.py`), 2. Array Utilities (`core/array_utils.py`), 3. Typed Fitting Context (`core/fit_context.py`), 4. Solver Abstraction Base Class (`fit/base.py`), 5. Fitting Workflow Orchestration (`core/fit_orchestrator.py`), Backward Compatibility, Code Quality Metrics, Code Refactoring Summary (+11 more)

### Community 72 - "test_step_no_freeze.py"
Cohesion: 0.83
Nodes (3): _run(), test_modern_trf_no_freeze(), test_modern_vp_no_freeze()

### Community 73 - "test_unc_label_aliases.py"
Cohesion: 0.67
Nodes (3): parametrize, test_canonical_unc_label_aliases(), test_canonical_unc_label_unknown()

### Community 74 - "test_unc_stats_keys_consistent.py"
Cohesion: 0.83
Nodes (3): _assert_param_block_has_consistent_keys(), _linear_fixture(), test_stats_keys_consistent_between_bootstrap_and_bayesian()

### Community 75 - "helptext.py"
Cohesion: 0.50
Nodes (3): build_help(), Detailed help text builder for Peakfit 3.x., Return an exhaustive help message. Parameters ---------- opts: Dictionary of…

### Community 77 - "Interactive Peak Fit GUI - Complete Codebase Documentation"
Cohesion: 0.12
Nodes (15): 13. Configuration, 17. File Structure with Descriptions, 1. Project Overview, 2. Architecture, 4. Data Flow, 6. GUI/UI Structure, Batch Workflow, Dependency Graph (+7 more)

### Community 78 - "5. Key Modules"
Cohesion: 0.12
Nodes (16): 5. Key Modules, `batch/runner.py` (1,315 lines), `core/data_io.py` (2,500+ lines), `core/fit_api.py` (850+ lines), `core/models.py` (148 lines), `core/peaks.py` (29 lines), `core/residuals.py` (184 lines), `core/signals.py` (116 lines) (+8 more)

### Community 82 - "test_smoke_gui_vs_batch_unc.py"
Cohesion: 0.28
Nodes (14): Serialize ``peaks`` into dictionaries suitable for JSON/CSV export., serialize(), finite_diff_jacobian(), Forward-difference Jacobian of ``residual_fn`` evaluated at ``theta``., _compare_rows(), _fit_once_gui_style(), _make_spectrum(), _predict_full_from_peaks() (+6 more)

### Community 83 - "run_batch"
Cohesion: 0.18
Nodes (12): _norm_jitter(), Any, Convenience wrapper used by smoke tests. Parameters mirror :func:`run` but…, Run the peak fitting pipeline over matching files. Parameters ----------…, Robust BLAS/threadpool limiter for batch path. Returns a real context object…, Normalize bootstrap jitter to a fractional value., run_batch(), run_from_dir() (+4 more)

### Community 84 - "Performance Apply Debounce Pattern - Knowledge Base"
Cohesion: 0.15
Nodes (12): 1. Scheduler/Debounce Infrastructure, 2. Request Method, 3. Signature Computation, 4. Core Apply with Logging Dedupe, 5. Trigger Wiring, Benefits Achieved, Common Pitfalls to Avoid, Critical Lesson Learned (+4 more)

### Community 85 - "15. Important Design Patterns"
Cohesion: 0.18
Nodes (11): 10. **CSV Schema Unification**, 15. Important Design Patterns, 1. **Lock-Aware Parameter Packing**, 2. **Dual Residual/Jacobian Functions**, 3. **Solver Dispatch via Options Dict**, 4. **Mode Isolation (Add vs Subtract)**, 5. **Uncertainty Result Container**, 6. **Defensive Backend Switching** (+3 more)

### Community 86 - "7. Key Architectural Patterns"
Cohesion: 0.18
Nodes (11): 10. CSV Schema Unification, 1. Lock-Aware Parameter Packing, 2. Dual Residual/Jacobian Functions, 3. Mode Isolation (Add vs Subtract), 4. Solver Dispatch via Options Dict, 5. Uncertainty Result Container, 6. Defensive Backend Switching, 7. Key Architectural Patterns (+3 more)

### Community 87 - "Interactive Peak Fit GUI - Comprehensive Codebase Analysis"
Cohesion: 0.20
Nodes (9): 11. Configuration Persistence, 2. Project Organization & Directory Structure, 6.1 Single-File Fitting Workflow, 6.2 Batch Processing Workflow, 6. Data Flow & Workflows, 8. Solver Comparison & Selection, Executive Summary, Interactive Peak Fit GUI - Comprehensive Codebase Analysis (+1 more)

### Community 88 - "Issue: Missing Import Error in Type Annotations"
Cohesion: 0.20
Nodes (9): Best Practices for Type Annotations in Tkinter Modules, Files Affected by This Pattern, Issue: Missing Import Error in Type Annotations, Lesson Learned, Problem Description, Root Cause, Solution Applied, Tkinter Font Rendering / Antialiasing Improvements - Knowledge Base (+1 more)

### Community 89 - "7. Fitting System"
Cohesion: 0.22
Nodes (9): 7. Fitting System, Bounds, Classic (SciPy `curve_fit`), LMFIT-VP (optional), Locks, Modern TRF (SciPy `least_squares`), Modern VP (Variable-Projection), Parameter Packing (+1 more)

### Community 90 - "design_matrix"
Cohesion: 0.28
Nodes (9): design_matrix(), eval_components(), eval_total(), ndarray, Handle shadow-compare mismatch by logging and falling back to NumPy., Evaluate the sum of pseudo-Voigt peaks on ``x``., Return component matrix with shape ``(n_peaks, len(x))``., Return design matrix ``A`` with columns per peak (shape ``m×n``). (+1 more)

### Community 91 - ".current_fit_mask"
Cohesion: 0.25
Nodes (3): ndarray, Return PSD covariance via SVD of the Jacobian., Clip negatives to zero before sqrt to avoid warnings.

### Community 92 - "test_progress_logging.py"
Cohesion: 0.29
Nodes (3): _AbortEvent, _DummySampler, test_bayesian_progress_logging_accepts_total_steps()

### Community 93 - "CLAUDE Memory — Interactive Peak Fit GUI"
Cohesion: 0.29
Nodes (6): Adding new documentation, CLAUDE Memory — Interactive Peak Fit GUI, Documentation, graphify, What this is, Where to look first

### Community 94 - "Export Formats"
Cohesion: 0.29
Nodes (7): 10. Export/Import, A) Peak Table (`*_fit.csv`), B) Trace CSV (`*_trace.csv`), C) Uncertainty CSV (`*_uncertainty.csv`), D) Uncertainty TXT (`*_uncertainty.txt`), Export Formats, Import Formats

### Community 95 - "blas_limit_ctx"
Cohesion: 0.29
Nodes (7): blas_limit_ctx(), blas_single_thread_ctx(), _log(), note_tpctl_absence_once(), Limit BLAS/OpenMP libraries to a single thread within the context., Limit BLAS/OpenMP threadpools within the scope. None or <=0 ⇒ no clamp (leave…, Log a single debug note when threadpoolctl is unavailable.

### Community 96 - "test_poly_baseline_caching.py"
Cohesion: 0.33
Nodes (5): cache_baseline_enabled(), set_cache_baseline(), test_cache_flag_controls_calls(), compute_cached(), test_poly_baseline_caching()

### Community 97 - "test_unc_bootstrap_outputs.py"
Cohesion: 0.38
Nodes (6): _maybe_read_unc_files(), _pivot_long_to_wide(), Path, Return (wide_df, long_df, used_path) where either or both can be None. Finds…, Convert long schema: file, peak, param, value, stderr, ci_lo, ci_hi, method,…, test_unc_bootstrap_outputs()

### Community 98 - "test_batch_runner.py"
Cohesion: 0.67
Nodes (6): _base_batch_cfg(), Path, Peak, test_run_batch_bootstrap_missing_samples_errors(), test_run_batch_bootstrap_requires_explicit_samples(), _write_sample()

### Community 100 - "11. Performance/Optimization"
Cohesion: 0.33
Nodes (6): 11. Performance/Optimization, Backends, Defensive Layer (`infra/performance.py`), Optimization Strategies, Seeding & Reproducibility, Threading & Multiprocessing

### Community 101 - "16. Known Limitations"
Cohesion: 0.33
Nodes (6): 16. Known Limitations, Batch, GUI, Mathematical, Solver, Uncertainty

### Community 102 - "Three Methods"
Cohesion: 0.33
Nodes (6): 8. Uncertainty Calculation, Asymptotic (Default Band ON), Band Construction, Bayesian (MCMC, optional emcee), Bootstrap (Residual Resampling), Three Methods

### Community 103 - "15. Known Limitations"
Cohesion: 0.33
Nodes (6): 15. Known Limitations, Batch, GUI, Mathematical, Solver, Uncertainty

### Community 104 - "TestThemeWithDefaultTkinter"
Cohesion: 0.33
Nodes (4): Test app functionality with default Tkinter (no themes)., Test that Tk root window can be created and destroyed., Test that ttk widgets work without custom themes., TestThemeWithDefaultTkinter

### Community 105 - "14. Dependencies"
Cohesion: 0.40
Nodes (5): 14. Dependencies, Core Dependencies, Development/Testing, Optional Features, Optional Performance

### Community 106 - "3. Core Functionality"
Cohesion: 0.40
Nodes (5): 3. Core Functionality, Baseline Correction, Fitting Modes, Interactive Workflows, Peak Model

### Community 107 - "9. Batch Processing"
Cohesion: 0.40
Nodes (5): 9. Batch Processing, Configuration, Output Files, Peak Seeding Strategies, Workflow

### Community 108 - "10. Export Schemas"
Cohesion: 0.40
Nodes (5): 10. Export Schemas, A) Peak Table (`_fit.csv`), B) Trace CSV (`_trace.csv`), C) Uncertainty CSV (`_uncertainty.csv`), D) Uncertainty TXT (`_uncertainty.txt`)

### Community 109 - "16. How to Extend the Codebase"
Cohesion: 0.40
Nodes (5): 16. How to Extend the Codebase, Adding a New Solver, Adding Baseline Method, Adding Uncertainty Method, Customizing UI

### Community 110 - "3. Core Technologies & Dependencies"
Cohesion: 0.40
Nodes (5): 3. Core Technologies & Dependencies, Development/Testing, Optional Features, Optional Performance, Primary Dependencies

### Community 111 - "test_bootstrap_workers.py"
Cohesion: 0.40
Nodes (4): Record a preferred seed for deterministic ops without globally seeding NumPy., set_max_workers(), set_seed(), test_bootstrap_parallel_deterministic()

### Community 112 - "test_template_auto_apply.py"
Cohesion: 0.40
Nodes (4): skip_if_no_display, Template auto-apply behaviour in GUI and batch modes., test_batch_template_auto_apply(), test_gui_template_auto_apply()

### Community 113 - "test_solver_parity.py"
Cohesion: 0.70
Nodes (4): parametrize, _reference_total(), _rmse(), test_solver_parity()

### Community 114 - "test_solver_regression.py"
Cohesion: 0.70
Nodes (4): _relative_diff(), test_lmfit_vp_regression(), test_modern_trf_regression(), test_modern_vp_regression()

### Community 115 - "12. Testing"
Cohesion: 0.50
Nodes (4): 12. Testing, Key Test Fixtures, Structure (164 test files, 778K), Testing Utilities

### Community 116 - "12. Testing Framework"
Cohesion: 0.50
Nodes (4): 12. Testing Framework, Test Categories, Test Fixtures, Testing Utilities

### Community 117 - "13. Performance Optimization"
Cohesion: 0.50
Nodes (4): 13. Performance Optimization, Backend Switching, Optimization Strategies, Threading & Multiprocessing

### Community 118 - "9. Uncertainty Methods"
Cohesion: 0.50
Nodes (4): 9. Uncertainty Methods, Asymptotic (Fast, Always Available), Bayesian (MCMC, Optional emcee), Bootstrap (Distribution-Free, Captures Nonlinearity)

### Community 119 - "build_peak_table"
Cohesion: 0.50
Nodes (3): build_peak_table(), Return a CSV-formatted peak table built from ``records``. ``records`` should…, test_build_peak_table_includes_baseline_metadata_headers()

### Community 120 - "TestRequirementsUpdated"
Cohesion: 0.50
Nodes (3): Test that requirements.txt is properly updated., Verify requirements.txt includes theme packages as optional., TestRequirementsUpdated

### Community 121 - "TestThemeImport"
Cohesion: 0.50
Nodes (3): Test that theme module can be imported., Verify ui.theme module exists and imports correctly., TestThemeImport

### Community 122 - "14. Recent Refactoring Improvements"
Cohesion: 0.67
Nodes (3): 14. Recent Refactoring Improvements, Completed Refactorings, Impact

### Community 123 - "1. Project Purpose & Overview"
Cohesion: 0.67
Nodes (3): 1. Project Purpose & Overview, Core Functionality, Key Features

### Community 124 - "4. Mathematical Foundation"
Cohesion: 0.67
Nodes (3): 4. Mathematical Foundation, Fitting Models, Pseudo-Voigt Peak Model

## Knowledge Gaps
- **253 isolated node(s):** `NotAvailable`, `What this is`, `graphify`, `Where to look first`, `Documentation` (+248 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PeakFitApp` connect `PeakFitApp` to `UncertaintyResult`, `._update_bootstrap_visibility`, `test_theme_application.py`, `.refresh_plot`, `save_config`, `._compute_uncertainty_sync`, `.current_fit_mask`, `._build_ui`, `pseudo_voigt`, `test_template_auto_apply.py`, `app.py`, `main`, `._apply_performance_now`, `test_log_buffer.py`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `Peak` connect `Peak` to `ensure_unc_common`, `BaselineCompute`, `PeakFitSolver`, `runner.py`, `test_z_audit.py`, `pv_sum`, `step_engine.py`, `test_step_parity_trf_vp_lmfit.py`, `peaks.py`, `pack_theta_bounds`, `signals.py`, `write_uncertainty_txt`, `bootstrap.py`, `predict_full`, `lmfit_backend.py`, `residuals.py`, `build_residual`, `modern_vp.py`, `param_utils.py`, `modern.py`, `build_trace_table`, `test_batch_bootstrap_band_toggle.py`, `test_step_no_freeze.py`, `test_smoke_gui_vs_batch_unc.py`, `run_batch`, `test_batch_runner.py`, `test_bootstrap_workers.py`, `test_template_auto_apply.py`, `test_solver_parity.py`, `test_solver_regression.py`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `run_batch()` connect `run_batch` to `ensure_unc_common`, `performance.py`, `._compute_uncertainty_sync`, `bootstrap_ci`, `runner.py`, `Any`, `route_uncertainty`, `data_io.py`, `peaks.py`, `Peak`, `signals.py`, `write_uncertainty_txt`, `UncertaintyResult`, `bootstrap.py`, `predict_full`, `residuals.py`, `build_residual`, `build_trace_table`, `test_batch_baseline_method_column.py`, `test_batch_bootstrap_band_toggle.py`, `write_batch_uncertainty_long`, `test_smoke_gui_vs_batch_unc.py`, `test_progress_logging.py`, `test_batch_runner.py`, `build_peak_table`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `PeakFitApp` (e.g. with `test_export_fit_columns()` and `test_export_uncertainty_single()`) actually correct?**
  _`PeakFitApp` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 72 inferred relationships involving `Peak` (e.g. with `lmfit_step()` and `modern_vp_step()`) actually correct?**
  _`Peak` has 72 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_batch()` (e.g. with `UncertaintyResult` and `.is_set()`) actually correct?**
  _`run_batch()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `NotAvailable`, `What this is`, `graphify` to the rest of the system?**
  _253 weakly-connected nodes found - possible documentation gaps or missing edges._