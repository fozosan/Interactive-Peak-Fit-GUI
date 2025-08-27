"""Core utilities for Peakfit 3.x.

This package exposes the main core modules so that callers can simply
``from core import data_io, models, peaks, residuals, signals`` without
needing to know the submodule structure. Importing here also helps static
analyzers resolve these symbols.
"""

from . import data_io, models, peaks, residuals, signals, uncertainty

__all__ = ["data_io", "models", "peaks", "residuals", "signals", "uncertainty"]
