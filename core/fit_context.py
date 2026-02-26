"""Typed context container for fitting operations.

This module provides a replacement for the previously-used fit_ctx dictionary,
ensuring type safety and reducing defensive null checks throughout the code.

The FitContext dataclass serves as a central repository for all state
information needed during fitting, uncertainty calculation, and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class FitContext:
    """Typed context for fitting operations.

    This dataclass consolidates all fitting state that was previously
    scattered across dictionary keys. It provides type safety and
    validation while maintaining backward compatibility with code that
    expects a dictionary-like interface.

    Attributes
    ----------
    add_mode : bool
        Whether fitting in "add" (baseline + peaks) vs "subtract" mode.
    solver : str
        Name of the solver used (e.g., "classic", "modern", "modern_vp").
    theta0 : ndarray
        Initial parameter vector.
    bounds : tuple[ndarray, ndarray]
        (lower, upper) bounds for parameters.
    peaks_out : list
        Fitted Peak objects.
    vp_etas : list[float]
        Shape factors (eta) for each peak.
    vp_unpack_cw : callable
        Function to unpack centers/widths from nonlinear vector.
    x_all : ndarray
        Full x data (all points).
    x_fit : ndarray
        X data for fit window only.
    y_target_fit : ndarray
        Target y data for fit window (baseline-subtracted if "add").
    wvec_fit : ndarray, optional
        Noise weights for fit window.
    base_all : ndarray, optional
        Baseline values for all points.
    mask : ndarray
        Boolean mask indicating fit window.
    predict_full : callable
        Function to predict y values for full x domain.
    locked_mask : ndarray, optional
        Boolean mask indicating locked parameters.
    share_fwhm : bool, default=False
        Whether all peaks share same FWHM value.
    share_eta : bool, default=False
        Whether all peaks share same eta (shape) value.
    bootstrap_jitter : float, optional
        Jitter magnitude for bootstrap resampling.
    unc_workers : int, default=1
        Number of workers for uncertainty calculation.
    unc_band_workers : int, default=1
        Number of workers for band calculation.
    unc_boot_solver : str, optional
        Solver override for bootstrap (may differ from main solver).
    bayes_sampler_cfg : dict, optional
        Configuration for Bayesian MCMC sampler.
    abort_event : object, optional
        Cancellation event for long operations.
    refit : bool, default=False
        Whether to refit in uncertainty calculations.
    strict_refit : bool, default=False
        Whether refit must use exact solver (no fallback).
    """

    # Core fitting state
    add_mode: bool
    solver: str
    theta0: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    peaks_out: List[Any]
    vp_etas: List[float]
    vp_unpack_cw: Callable
    x_all: np.ndarray
    x_fit: np.ndarray
    y_target_fit: np.ndarray

    # Data and weighting
    wvec_fit: Optional[np.ndarray] = None
    base_all: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    # Prediction
    predict_full: Optional[Callable] = None

    # Parameter constraints
    locked_mask: Optional[np.ndarray] = None
    share_fwhm: bool = False
    share_eta: bool = False

    # Uncertainty configuration
    bootstrap_jitter: Optional[float] = None
    unc_workers: int = 1
    unc_band_workers: int = 1
    unc_boot_solver: Optional[str] = None
    bayes_sampler_cfg: Optional[Dict[str, Any]] = None

    # Control flow
    abort_event: Optional[Any] = None
    refit: bool = False
    strict_refit: bool = False

    # Custom metadata (for extensibility)
    _extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access for backward compatibility.

        Parameters
        ----------
        key : str
            Attribute name.

        Returns
        -------
        Any
            Attribute value.

        Raises
        ------
        KeyError
            If key is not found.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            if key in self._extra:
                return self._extra[key]
            raise KeyError(f"FitContext has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like assignment for backward compatibility.

        Parameters
        ----------
        key : str
            Attribute name or custom key.
        value : Any
            Value to set.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback.

        Parameters
        ----------
        key : str
            Attribute name.
        default : Any
            Default if key not found.

        Returns
        -------
        Any
            Attribute value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, Any]
            All attributes as a dictionary.
        """
        data = asdict(self)
        # Remove _extra and fold into main dict
        extra = data.pop("_extra", {})
        data.update(extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FitContext:
        """Create instance from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with keys matching FitContext attributes.

        Returns
        -------
        FitContext
            New instance.
        """
        # Extract known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known_fields.discard("_extra")

        kwargs = {}
        extra = {}

        for key, value in data.items():
            if key in known_fields:
                kwargs[key] = value
            else:
                extra[key] = value

        kwargs["_extra"] = extra
        return cls(**kwargs)

    def validate(self) -> None:
        """Validate context consistency.

        Raises
        ------
        ValueError
            If required fields are missing or inconsistent.
        """
        if self.theta0 is None:
            raise ValueError("theta0 is required")
        if self.bounds is None or len(self.bounds) != 2:
            raise ValueError("bounds must be (lower, upper) tuple")
        if self.x_fit is None or self.y_target_fit is None:
            raise ValueError("x_fit and y_target_fit are required")
        if len(self.peaks_out) != len(self.vp_etas):
            raise ValueError("peaks_out and vp_etas length mismatch")
        if self.wvec_fit is not None:
            if len(self.wvec_fit) != len(self.x_fit):
                raise ValueError("wvec_fit length must match x_fit")
