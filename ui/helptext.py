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
        ==================================================

        Quick start
        -----------
        1. Open a 2-column file (x, y). Optionally pick a fit window.
        2. Compute the ALS baseline (defaults usually work).
        3. Add peaks: click on the plot or auto-seed.
        4. Choose a solver (Classic / Modern TRF / Modern VP / LMFIT-VP if installed).
        5. Fit. Use Step ▶ to watch single iterations. Export peaks/trace CSVs.

        Data loading
        ------------
        • “Open Data…” reads CSV/TXT/DAT with 2 numeric columns (x, y).
        • Delimiters auto-detected; lines with #, %, // (or text headers) are ignored.
        • Non-finite rows are dropped. If x is descending, it is sorted ascending.

        Baseline (ALS)
        --------------
        • λ (smooth): larger ⇒ smoother baseline (stiffer).
        • p (asym): pushes baseline underneath peaks (smaller p hugs under peaks).
        • Iterations: maximum IRLS passes. Threshold: early-stop when Δbaseline is tiny.
        • “Baseline uses fit range”: estimate z(x) only in the selected window, then
          interpolate/extrapolate across full x (constant beyond ends).
        • Modes:
          – Add:    model = baseline + Σ peaks, residuals compared to raw y.
          – Subtract: model = Σ peaks, residuals compared to y − baseline.

        Fit window
        ----------
        • Enter Min/Max or “Select on plot” and drag. “Full range” clears it.
        • The shaded region shows the active window (affects ALS if enabled).

        Peaks: add, edit, lock
        ----------------------
        • “Add peaks on click”: click at the desired center. Height is initialized from
          the (optionally) baseline-corrected signal near the click.
        • Edit per-peak Center / Height / FWHM. Lock width and/or center to keep them fixed.
        • Shape factor η (0=Gaussian, 1=Lorentzian). Use “Apply to all” to broadcast η.
        • “Auto-seed” finds prominent peaks within the active window.

        Solvers — when to use what
        ---------------------------
        Classic (curve_fit):
          • Simple, unweighted least-squares with minimal bounds; respects locks.
          • Best for clean spectra or quick checks. Fast and interpretable.

        Modern TRF (robust least_squares):
          • For outliers, spikes, or heteroscedastic noise. Supports robust loss and weights.
          • Loss choices: {modern_losses}.
          • Noise weights: {modern_weights}.
          • Under the hood: SciPy’s trust-region reflective algorithm with bounds. We premultiply
            residuals/Jacobian by noise weights; robust loss is handled by the solver’s M-estimator.

        Modern VP (variable projection):
          • For overlapped peaks and stability. Splits parameters into linear (heights) and
            nonlinear (centers/widths).
          • At each iterate, heights are solved by **non-negative least squares (NNLS)** on the
            current design matrix; then centers/widths take a Gauss–Newton step with backtracking.
          • Jacobians (∂P/∂center, ∂P/∂width) use stable finite-difference columns on unit-height shapes.

        LMFIT-VP (optional, if lmfit installed):
          • A thin wrapper using lmfit’s least_squares backend with the same packing/locks. Algorithms: {lmfit_algos}.

        Step ▶ (single iteration)
        -------------------------
        • Runs exactly one solver update with the same residuals/weights/bounds used by Fit.
        • Reports damping λ, backtracks, step_norm, acceptance, and reason.
        • Tips: If steps are rejected often, decrease λ, improve initial guesses, or run a full Fit to re-linearize.

        Uncertainty (current: asymptotic)
        ---------------------------------
        • Computes the Jacobian J at the solution on the fit window. With RSS = ||r||² and dof = m−n:
            σ² = RSS / dof, Cov(θ) ≈ σ² (JᵀJ)⁻¹  (tiny Tikhonov if needed).
          A 95% CI band for ŷ(x) uses the delta method: Var[ŷ] = diag(G Cov Gᵀ) with G = ∂ŷ/∂θ.
        • Bootstrap/MCMC are planned for future versions.

        Batch processing
        ----------------
        • Choose folder + patterns (semicolon-separated), e.g., *.csv;*.txt;*.dat
        • Peaks source: Current | Selected template | Auto-seed. “Re-height per file” adapts heights from each file’s signal.
        • Writes a summary CSV plus optional per-spectrum trace CSVs. Progress and messages stream to the status/log panel.

        Exports (schemas are fixed across single & batch)
        -------------------------------------------------
        Peak table CSV columns (fixed order):
          file, peak, center, height, fwhm, eta, lock_width, lock_center,
          area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax
          – area: closed-form pseudo-Voigt area (Gaussian+Lorentzian mix)
          – area_pct: 100 × area / Σ area
          – rmse: computed on the active window against the correct target (Add: raw; Subtract: raw − baseline)

        Trace CSV columns (fixed order):
          x, y_raw, baseline,
          y_target_add, y_fit_add, peak1, peak2, …,
          y_target_sub, y_fit_sub,  peak1_sub, peak2_sub, …
          – peakN     = baseline-ADDED curve (for WYSIWYG plotting in Add mode)
          – peakN_sub = baseline-SUBTRACTED pure component (for calculations)

        Status bar & log (what you’ll see)
        ----------------------------------
        • Progress indicator: shows long tasks (batch, uncertainty).
        • Log console: collapsible panel with per-step messages, solver diagnostics, and batch outcomes.

        Keyboard & mouse
        ----------------
        • Zoom/Pan tools temporarily disable click-to-add (so zooming never adds peaks).
        • Mouse-wheel over the right panel scrolls that panel; the Help window scrolls independently.
        • “Reset view” and “Zoom out” help you navigate quickly.

        Persistence
        -----------
        • The app remembers: global η, “Add peaks on click”, solver selection, and uncertainty method,
          plus ALS defaults, batch defaults, x-label, and templates. Stored in ~/.gl_peakfit_config.json.

        Numerical notes (for the curious)
        ---------------------------------
        • Pseudo-Voigt: g(x; h,c,w,η) = h * [(1−η) * exp(−4 ln 2 ((x−c)/w)²) + η / (1 + 4((x−c)/w)²)].
        • Variable projection (VP): build A with unit-height peak columns; solve h ≥ 0 via NNLS on W(Ah − y).
          Then step (c,w) using a Gauss–Newton update on the reduced objective; halve/backtrack if cost increases.
        • TRF robust loss: {modern_losses}. Noise weights: {modern_weights}.
          Poisson ≈ 1/√max(|y|, ε). inv_y ≈ 1/max(|y|, ε). Residuals/Jacobians are premultiplied by W.

        Troubleshooting
        ---------------
        • Fit looks too tall in Add mode → verify you intended “Add” (baseline is included) vs “Subtract”.
        • ALS hugging peak tops → increase λ and/or reduce p; enable “baseline uses fit range”.
        • Step ▶ keeps getting rejected → reduce λ, refine initial peaks, or run a full Fit to re-linearize.
        • Heavy tails/outliers → use Modern TRF with soft_l1 / huber / cauchy; try Poisson weights.
        • Overlapped peaks → Modern VP is often more stable (heights from NNLS).
        • Poor scaling → widen the fit window a touch or seed widths closer to visual FWHM.

        FAQ
        ---
        Q: Which solver should I start with?
           Classic for quick/clean cases, TRF for robustness, VP for overlapped peaks. LMFIT-VP if you prefer lmfit’s tooling.
        Q: Do locks apply to all solvers?
           Yes—locked centers/widths are fixed throughout Fit and Step ▶.
        Q: Can I see uncertainty?
           Yes—choose “Asymptotic” and run uncertainty to overlay a 95% CI band.

        Credits
        -------
        Designed by Farhan Zahin • Built with ChatGPT (v3.2-beta)
        """
    ).strip()
