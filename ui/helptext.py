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
Interactive GL (pseudo-Voigt) Peak Fitting — Help
==================================================

1) Load data
• “Open Data…” accepts 2-column x,y (CSV/TXT/DAT). Delimiters auto-detected. Lines with # % // or text headers are ignored.
• Data is sorted if x is descending and non-finite values are dropped.

2) Baseline (ALS)
• λ (smooth): larger = smoother baseline.   • p (asym): pushes baseline under peaks.
• Iterations: maximum ALS passes.           • Threshold: early-stop when updates are tiny.
• “Baseline uses fit range”: compute ALS in the selected x-window then interpolate across the full x.
• Modes:
  – Add to fit  : model = baseline + Σ(peaks) (components plotted baseline-added)
  – Subtract    : model = Σ(peaks) against (y − baseline)

3) Fit range
• Enter Min/Max or “Select on plot” (drag). “Full range” clears. Shaded region = active fit window.

4) Peaks
• Toggle “Add peaks on click”, then click at the desired center. Height is taken from the (optionally) baseline-corrected signal.
• Edit Center/Height/FWHM in the table; lock Width and/or Center to hold them fixed during fitting.
• Shape factor η: 0=Gaussian, 1=Lorentzian. “Apply to all” broadcasts the current η to every peak.
• “Auto-seed” finds prominent peaks inside the active window.

5) Fitting methods
• Classic (curve_fit): simple unweighted least-squares with minimal bounds; honors locks; best for clean spectra.
• Modern TRF: robust least-squares with bounds; choose loss (linear/soft_l1/huber/cauchy) and optional weights (none/poisson/inv_y).
• Modern VP: variable-projection—heights solved quickly; robust loss/weights supported.
• LMFIT-VP (optional): available if lmfit is installed.
• “Step ▶” performs a single damped Gauss–Newton/TRF update using the same residuals/bounds as Fit and only commits parameters when cost decreases.

6) Uncertainty
• Asymptotic: computes covariance from the Jacobian at the solution and overlays a 95% CI band.
• Bootstrap/MCMC planned for future releases.

7) Batch
• Patterns like *.csv;*.txt;*.dat (semicolon-separated).
• Peaks source: Current | Selected template | Auto-seed. Optional re-height per file.
• Writes one summary CSV and (optionally) a per-spectrum trace CSV.

8) Exports
• Peak table CSV columns:
  file, peak, center, height, fwhm, eta, lock_width, lock_center, area, area_pct, rmse, fit_ok, mode, als_lam, als_p, fit_xmin, fit_xmax
• Trace CSV contains both “added” and “subtracted” sections for downstream analysis.

9) Tips & troubleshooting
• If fits look too tall in Add mode, confirm Add/Subtract is what you intend.
• If ALS rides on peaks, increase λ and/or lower p. For spiky data use robust loss (soft_l1/huber/cauchy) and consider Poisson weights.
• If Step is rejected, try a smaller λ or run Fit to re-linearize.
• The right panel scrolls with the mouse wheel when your cursor is over it; the Help window has its own scroll.

10) Persistence
• η, “Add peaks on click”, solver choice, and uncertainty method persist across sessions. Change them, and they’ll be remembered on next launch.
        """
    ).strip()

