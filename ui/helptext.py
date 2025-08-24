"""Help text for Peakfit 3.x panels."""
from __future__ import annotations

HELP = {
    "intro": (
        "Interactive Peak Fit 3.x\n\n"
        "Load a CSV/TXT/DAT spectrum, adjust the baseline, add peaks and run the solver."
    ),
    "baseline": (
        "Baseline (ALS)\n"
        "  • λ – smoothness; higher is smoother\n"
        "  • p – asymmetry (0 symmetric)\n"
        "  • Iterations and Threshold control convergence\n"
        "  • 'Baseline uses fit range' restricts ALS to the current x-range\n"
        "  • Recompute baseline after changing settings; Save defaults for next time"
    ),
    "eta": (
        "Shape factor η\n"
        "  • 0 = Gaussian, 1 = Lorentzian\n"
        "  • 'Apply to all peaks' pushes the value to every peak"
    ),
    "peaks": (
        "Peaks table\n"
        "  • Select a peak to edit Center/Height/FWHM\n"
        "  • Lock width/center to fix during fitting\n"
        "  • Use the buttons to add, apply edits, delete or clear peaks"
    ),
    "interaction": (
        "Interaction\n"
        "  • 'Add peaks on click' toggles click-to-add\n"
        "  • Zoom out or Reset view to change the plot window"
    ),
    "fit_range": (
        "Fit range\n"
        "  • Type Min/Max or select on plot\n"
        "  • 'Full range' clears the limits"
    ),
    "templates": (
        "Peak templates\n"
        "  • Save peak sets and re-apply later\n"
        "  • 'Auto-apply on open' loads the selected template with new data"
    ),
    "performance": (
        "Performance options\n"
        "  • Numba/CuPy accelerate calculations\n"
        "  • Cache baseline saves recomputation\n"
        "  • Deterministic seed enables repeatable random numbers"
    ),
    "solver": (
        "Solver selection\n"
        "  • Classic – traditional least squares\n"
        "  • Modern – step engine\n"
        "  • LMFIT – uses the lmfit library"
    ),
    "step": (
        "Step ▶\n"
        "  • Runs a single solver iteration for diagnostics"
    ),
    "actions": (
        "Actions\n"
        "  • Auto-seed attempts to guess peaks\n"
        "  • Fit runs the solver to convergence\n"
        "  • Toggle components shows/hides individual peaks"
    ),
    "uncertainty": (
        "Uncertainty\n"
        "  • Choose method and number of iterations/steps\n"
        "  • Run computes uncertainties; Export saves results"
    ),
}
