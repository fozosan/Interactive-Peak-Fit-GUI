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
        ``lmfit_algos``.
    """

    modern_losses = ", ".join(opts.get("modern_losses", []))
    modern_weights = ", ".join(opts.get("modern_weights", []))
    lmfit_algos = ", ".join(opts.get("lmfit_algos", []))

    return dedent(
        f"""
        Interactive Peak Fit 3.x – Help
        =================================

        Data / File
        ------------
        • Open CSV, TXT or DAT spectra with two numeric columns. Delimiters are
          detected automatically and lines starting with #, %, or // are
          skipped.
        • Export saves a peak table and a full trace CSV matching the v2.7
          schema.

        Baseline (ALS)
        --------------
        • 'Apply baseline' toggles ALS correction with parameters λ (smooth),
          p (asymmetry), number of iterations and an early-stop threshold.
        • 'Baseline uses fit range' computes the baseline inside the current
          window then interpolates across the full x range.
        • Modes: Add (fit baseline + peaks to raw data) or Subtract (fit peaks
          to baseline‑subtracted data).

        Fit Range
        ---------
        • Enter Min/Max or drag on the plot using 'Select on plot'.
        • 'Full range' clears the limits.

        Peaks
        -----
        • Global η sets the Gaussian–Lorentzian mix; 'Apply to all' broadcasts
          the value to every peak.
        • The table lists center, height and FWHM with locks for width and
          center. '➕ Add Peak' inserts a peak using the typed fields.

        Solver
        ------
        • Classic (curve_fit) – simple unweighted least squares. Option: max evals.
        • Modern – Trust Region Reflective with robust losses
          [{modern_losses}], weighting [{modern_weights}], multi-start
          restarts and optional jitter. Toggles: 'Centers in window' and
          'Min FWHM ≈2×Δx'.
        • LMFIT – interface to the optional ``lmfit`` package with algorithms
          [{lmfit_algos}], plus 'Share FWHM' and 'Share η' constraints.

        Uncertainty
        -----------
        • Asymptotic – curvature of the final solution.
        • Bootstrap – residual, wild or pairs resampling with optional
          parallel workers and deterministic seeds.
        • Bayesian – MCMC via ``emcee`` with Gaussian, Student‑t or Poisson
          likelihoods.

        Performance
        -----------
        • Toggles for Numba JIT, GPU/CuPy acceleration, baseline caching,
          deterministic seeds and parallel bootstrap.
        • Configure global seed, max workers and GPU chunk size.

        Templates
        ---------
        • Save current peaks as named templates, apply them later and
          optionally auto-apply on open.

        Batch
        -----
        • Process folders of spectra using templates or current peaks with
          options to re-height and save traces.

        Actions & Diagnostics
        ---------------------
        • 'Step ▶' performs one Gauss–Newton iteration.
        • 'Fit' runs the selected solver.
        • The status bar shows S/N estimates and solver messages; residual or
          credible bands can be toggled when available.
        """
    ).strip()

