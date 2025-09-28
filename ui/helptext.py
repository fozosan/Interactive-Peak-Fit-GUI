"""Detailed help text builder for Peakfit 3.x."""
from __future__ import annotations

from textwrap import dedent


def build_help(opts: dict) -> str:
    """Return an exhaustive help message.

    Parameters
    ----------
    opts:
        Dictionary of option lists used to keep this help in sync with the UI.
        Expected keys include ``modern_losses``, ``modern_weights``, and
        ``lmfit_algos`` (optional).
    """
    def _list(key, fallback):
        vals = opts.get(key) or []
        return ", ".join(vals) if vals else fallback

    modern_losses  = _list("modern_losses",  "linear, soft_l1, huber, cauchy")
    modern_weights = _list("modern_weights", "none, poisson, inv_y")
    lmfit_algos    = _list("lmfit_algos",    "least_squares")

    return dedent(
        f"""
        Interactive GL (pseudo-Voigt) Peak Fitting — Help
        ================================================

        What this app does
        ------------------
        Fit 1D spectra with sums of pseudo-Voigt peaks on top of an ALS baseline.
        You can work interactively (single file) or batch a whole folder, and export
        a peak table, full traces, and uncertainty reports with consistent schemas.

        TL;DR Flow (beginner road map)
        ------------------------------
                         ┌─────────────┐
                         │  Open data  │  CSV/TXT/DAT with 2 columns (x,y)
                         └──────┬──────┘
                                ▼
                      ┌──────────────────┐
                      │ Pick fit window  │  drag on plot or type limits
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Compute baseline │  ALS (λ smooth, p asym); by default uses the window
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Add / auto-seed  │  click on plot, edit params, lock width/center as needed
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Choose solver    │  Classic / Modern TRF / Modern VP / LMFIT-VP*
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Step ▶ (optional)│  watch one iteration, tune if needed
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Fit              │  inspect residuals / RMSE
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Uncertainty      │  95% CI band (asymptotic/boot/Bayes†)
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Export           │  *_fit.csv, *_trace.csv, *_uncertainty.csv/txt
                      └──────────────────┘
                      * if lmfit installed  †Bayes requires emcee

        Quick start
        -----------
        1) Open a 2-column file (x, y). Lines with #, %, // and text headers are ignored.
        2) Select a fit window (drag on plot) — baseline uses the window by default.
        3) Click “Recompute baseline” (defaults are usually OK: larger λ = smoother; smaller p hugs under peaks).
        4) Add peaks (click), or “Auto-seed” within the window. Lock width/center where appropriate.
        5) Choose a solver:
           • Classic: fast, simple least-squares (good for clean data)
           • Modern TRF: robust loss + noise weights (outliers/heavy-tails)
           • Modern VP: stable on overlaps (heights via NNLS)
           • LMFIT-VP: optional if lmfit is present
        6) Fit. Optionally Step ▶ first to see one iteration.
        7) Run Uncertainty (defaults to Asymptotic; Bootstrap/Bayesian available if configured).
        8) Export: *_fit.csv, *_trace.csv, *_uncertainty.csv, *_uncertainty.txt (± values).

        Data loading
        ------------
        • Delimiters auto-detected (comma/tab/space/semicolon); non-numeric columns dropped.
        • Non-finite rows removed; x is sorted ascending if needed.

        Baseline (ALS)
        --------------
        • λ (smoothness): larger ⇒ smoother baseline.
        • p (asymmetry): pushes baseline below peaks; smaller p hugs under peaks.
        • Iterations: max IRLS passes. Threshold: early stop when Δz is tiny.
        • “Baseline uses fit range”: ON by default. Compute within the window, then interpolate across full x
          (constant beyond ends).
        • Modes:
          – Add     : model = baseline + Σ peaks; residuals vs raw y.
          – Subtract: model = Σ peaks; residuals vs (y − baseline).

        Baseline methods (ALS & Polynomial)
        -----------------------------------
        • Choosing a method: Baseline panel → Method: als or polynomial. “Save as default” persists
          the method and parameters; last used method is restored on next launch.
        • ALS (Asymmetric Least Squares): λ (smoothness), p (asymmetry), Iterations, Threshold
          (early stop). Optionally limit to the fit range, then interpolate to the full x-range.
        • Polynomial baseline: Degree (≥0) and optional “Normalize x to [-1,1]”. If the fit window
          has too few points, the degree is auto-clamped to min(requested, N−1); the UI updates the
          field and posts a status note.
        • Baseline uses fit range: When enabled, baseline is estimated only inside the current fit
          window and smoothly extended outside.
        • Exports: peak table includes baseline_method; ALS fields (als_*) are populated for ALS
          and NaN under polynomial; polynomial fields (poly_degree, poly_normalize_x) are populated
          for polynomial and NaN under ALS.

        Peaks: add, edit, lock
        ----------------------
        • “Add peaks on click” toggles interactive placement at the click x.
        • Edit Center / Height / FWHM. Lock width and/or center to keep them fixed.
        • Shape factor η per peak (0=Gaussian, 1=Lorentzian). “Apply to all” broadcasts η.
        • Auto-seed finds prominent peaks within the window (baseline-aware).

        Solvers — when to use what
        ---------------------------
        Classic (curve_fit):
          • Simple, unweighted least-squares with minimal bounds; respects locks.
          • Best for clean spectra or quick checks.

        Modern TRF (robust least_squares):
          • For outliers/heteroscedastic noise. Robust loss and per-point noise weights.
          • Loss: {modern_losses}
          • Weights: {modern_weights}  (poisson ≈ 1/√max(|y|,ε); inv_y ≈ 1/max(|y|,ε))
          • Uses SciPy’s trust-region reflective algorithm with bounds. Residuals & Jacobian are premultiplied by
            noise weights; robust loss is handled internally by the solver.

        Modern VP (variable projection):
          • For overlapped peaks and stability.
          • At each iterate: heights by **non-negative least squares (NNLS)** on the current design matrix; then
            centers/widths step via Gauss–Newton with backtracking. Bounds/locks respected.

        LMFIT-VP (optional):
          • Thin wrapper over lmfit with the same packing/locks. Algorithms: {lmfit_algos}

        Step ▶ (single iteration)
        -------------------------
        • Runs exactly one solver update with the SAME residuals/weights/bounds used by Fit.
        • Reports λ (damping), backtracks, step_norm, accept/reject, and reason.
        • If steps are rejected often: decrease λ, refine seeds (centers/FWHM), or run a full Fit to re-linearize.

        Uncertainty (bands & stats)
        ---------------------------
        • Asymptotic (default band ON): fast covariance-based CIs using JᵀJ; prediction band via delta method; lightly smoothed.
        • Bootstrap: residual resampling → refit → parameter distributions; seeded for reproducibility; respects locks/bounds.
        • Bayesian (emcee): posterior mean/SD and 95% credible intervals; optional diagnostics (ESS, R̂, MCSE) and posterior-predictive band.
          – ESS (effective sample size): higher is better; <200 suggests more steps/chains.
          – R̂ (“R-hat”): should be ≈1.00; >1.05 suggests non-convergence.
          – MCSE: Monte Carlo standard error of reported quantiles (q16/q50/q84); smaller is better.
          You can toggle diagnostics via “Compute diagnostics (ESS/R̂/MCSE)” to reduce overhead on large runs.
          (If emcee is not installed, the app reports “NotAvailable”.)

        Batch processing
        ----------------
        • Choose folder + patterns (semicolon-separated), e.g., *.csv;*.txt;*.dat
        • Peaks source: Current | Template (templates auto-apply) | Auto-seed
        • Options: re-height per file; per-spectrum trace exports; chosen output directory.
        • The batch runner deep-copies seeds, recomputes the baseline inside each file’s window, enforces locks/bounds,
          and matches single-file RMSE/parameters.

        Exports (single & batch share formats; no blank lines)
        ------------------------------------------------------
        • *_fit.csv        : peak table with solver/baseline/perf metadata
          Columns (fixed order):
            file, peak, center, height, fwhm, eta, lock_width, lock_center,
            area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax,
            solver_choice, solver_loss, solver_weight, solver_fscale, solver_maxfev,
            solver_restarts, solver_jitter_pct, step_lambda,
            baseline_uses_fit_range, perf_numba, perf_gpu, perf_cache_baseline,
            perf_seed_all, perf_max_workers
        • *_trace.csv      : full traces with BOTH sections:
            x, y_raw, baseline,
            y_target_add, y_fit_add, peak1, peak2, …,
            y_target_sub, y_fit_sub, peak1_sub, peak2_sub, …
          – peakN      = baseline-ADDED (matches Add-mode display)
          – peakN_sub  = baseline-SUBTRACTED (pure components)
        • *_uncertainty.csv: tabular parameter stats (mean, sd, CI) and optional band summaries
        • *_uncertainty.txt: human-readable report with “±” values and method notes
        • Batch also writes a summary CSV; all files go to the output folder you select.

        Action bar, legend, and log
        ---------------------------
        • Action bar: File [Open, Export, Batch] | Fit [Step, Fit] | Plot [Uncertainty, Legend, Components] | Help (F1).
        • Legend: toggle on/off; entries include peak centers; font is Arial.
        • Log: collapsible green-on-black console with solver/uncertainty/batch diagnostics.

        Persistence
        -----------
        • Saved in ~/.gl_peakfit_config.json: ALS defaults, solver choice & options, uncertainty method, click-to-add,
          global η, performance toggles (Numba/CuPy/cache/seed_all/max_workers), batch defaults, x-label, legend visibility,
          and templates (including auto-apply).

        Practical tips
        --------------
        • If fits look too tall, verify you intended Add vs Subtract mode.
        • If ALS rides peak tops, increase λ and/or decrease p; “baseline uses fit range” often helps.
        • For spiky residuals/heavy tails, switch to TRF with soft_l1/huber/cauchy and try Poisson weights.
        • For overlapped peaks, VP is often more stable (heights via NNLS).
        • If the CI band has “spikes”: lock weakly determined params, narrow the window, or increase bootstrap samples.

        Nerd corner (math & algorithms)
        -------------------------------
        Pseudo-Voigt:
            g(x; h,c,w,η) = h * [(1−η) * exp(−4 ln 2 * ((x−c)/w)^2) + η / (1 + 4((x−c)/w)^2)]
        Model:
            P(x; θ) = Σ_j g_j(x; h_j, c_j, w_j, η_j),   y_fit = baseline + P (Add)  or  y_fit = P (Subtract)
        Residuals:
            r = y_fit − y_target,   where y_target = y_raw (Add) or (y_raw − baseline) (Subtract)
        Weighting:
            W_noise = diag(w_i),  w_i ∈ {{ {modern_weights} }}
            TRF robust loss ∈ {{ {modern_losses} }} handled by SciPy’s M-estimator; we premultiply residuals/J by W_noise.
        ALS baseline:
            z = argmin ||W(y − z)||² + λ ||D² z||²,  W from p via IRLS. Solve (W + λ DᵀD + εI)z = Wy until Δz small.
        Variable projection (VP):
            A[:,j] = g_j(x; h=1, c_j, w_j, η_j).  Solve h ≥ 0 by NNLS on W(Ah − y).  Update (c,w) by Gauss–Newton on the
            reduced objective; step-halve (backtrack) if cost↑. Bounds/locks applied each step.
        Step ▶ (damped GN/LM):
            Solve (JᵀJ + λI)δ = −Jᵀr.  Accept if cost↓; otherwise increase λ or halve δ and retry (bounded backtracks).
        Asymptotic CI (delta method):
            σ² = RSS / dof,  Cov ≈ σ² (JᵀJ)⁻¹ (tiny Tikhonov if ill-conditioned),
            Var[ŷ] = diag(G Cov Gᵀ), 95% band = ŷ ± 1.96√Var. Bands lightly smoothed for display.
        Bootstrap CI:
            Residual resampling: y* = y_fit + r*; refit to get θ* draws → parameter stats and optional predictive band.
        Bayesian CI (emcee):
            Posterior over free params; report mean/SD/95% CI; check ESS and R-hat; predictive band from posterior draws.

        Troubleshooting
        ---------------
        • “Rank-deficient Jacobian” in uncertainty: some params are weakly identified. Lock them or narrow the window.
        • “Ill-conditioning” warning: try TRF with x_scale (default), or VP; seed widths closer to visual FWHM.
        • Single vs batch mismatch: ensure templates auto-apply and “baseline uses fit range” is ON (default). Our runner
          already deep-copies seeds and enforces locks/bounds.

        FAQ
        ---
        • Which solver should I try first?  Classic for clean data, TRF for robustness, VP for overlaps.
        • Do locks apply everywhere?       Yes, for Fit and Step ▶ across all solvers.
        • Can I reproduce results?         Yes—set a seed; batch uses per-file seeds and fixed worker caps.

        Credits
        -------
        Designed by Farhan Zahin • Built with ChatGPT (v3.2)
        """
    ).strip()
