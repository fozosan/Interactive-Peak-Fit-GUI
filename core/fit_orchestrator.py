"""Orchestration logic for fitting workflow.

This module encapsulates the orchestration and workflow management that was
previously embedded in run_fit_consistent(). It breaks down the complex
fitting pipeline into manageable, testable components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import copy
from dataclasses import replace

from .peaks import Peak
from .fit_context import FitContext
from . import signals
from .signals import polynomial_baseline
from fit.bounds import pack_theta_bounds


@dataclass
class BaselineConfig:
    """Configuration for baseline computation."""

    method: str = "als"
    lam: float = 1e5
    p: float = 0.001
    niter: int = 10
    thresh: float = 0.0
    degree: int = 2
    normalize_x: bool = True
    uses_fit_range: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> BaselineConfig:
        """Create from configuration dictionary."""
        return cls(
            method=str(cfg.get("method", "als")).strip().lower(),
            lam=float(cfg.get("lam", 1e5)),
            p=float(cfg.get("p", 0.001)),
            niter=int(cfg.get("niter", 10)),
            thresh=float(cfg.get("thresh", 0.0)),
            degree=int(cfg.get("degree", 2)),
            normalize_x=bool(cfg.get("normalize_x", True)),
            uses_fit_range=bool(cfg.get("uses_fit_range", True)),
        )


class BaselineCompute:
    """Computes baseline correction with optional windowing."""

    def __init__(self, config: BaselineConfig):
        """Initialize with baseline configuration.

        Parameters
        ----------
        config : BaselineConfig
            Baseline method and parameters.
        """
        self.config = config

    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute baseline for full x domain.

        Parameters
        ----------
        x : ndarray
            Full x values.
        y : ndarray
            Full y values.
        mask : ndarray, optional
            Boolean mask for fit window. If provided and method requires it,
            baseline is computed on window then interpolated to full domain.

        Returns
        -------
        ndarray
            Baseline values for all x points.

        Raises
        ------
        ValueError
            If baseline method is unknown.
        """
        method = self.config.method

        if method == "als":
            return self._compute_als(x, y, mask)
        elif method == "polynomial":
            return self._compute_polynomial(x, y, mask)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    def _compute_als(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute ALS baseline with optional windowing."""
        fit_mask_eff = mask if self.config.uses_fit_range and mask is not None else None

        if fit_mask_eff is not None and np.any(fit_mask_eff) and not np.all(fit_mask_eff):
            # Compute on fit window, interpolate to full domain
            x_sub = x[fit_mask_eff]
            y_sub = y[fit_mask_eff]
            z_sub = signals.als_baseline(
                y_sub,
                lam=self.config.lam,
                p=self.config.p,
                niter=self.config.niter,
                tol=self.config.thresh,
            )
            baseline = np.interp(x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
        else:
            # Compute on full domain
            baseline = signals.als_baseline(
                y,
                lam=self.config.lam,
                p=self.config.p,
                niter=self.config.niter,
                tol=self.config.thresh,
            )

        return baseline

    def _compute_polynomial(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute polynomial baseline."""
        fit_mask_eff = mask if self.config.uses_fit_range and mask is not None else None
        return polynomial_baseline(
            x,
            y,
            degree=self.config.degree,
            mask=fit_mask_eff,
            normalize_x=self.config.normalize_x,
        )


class PeakInitializer:
    """Handles initial peak adjustment (e.g., reheighting)."""

    def __init__(self, reheight: bool = False):
        """Initialize.

        Parameters
        ----------
        reheight : bool, default=False
            Whether to adjust initial peak heights based on data.
        """
        self.reheight = reheight

    def initialize(
        self,
        peaks: List[Peak],
        x: np.ndarray,
        y: np.ndarray,
        baseline: Optional[np.ndarray],
    ) -> List[Peak]:
        """Initialize peaks, optionally adjusting heights.

        Parameters
        ----------
        peaks : list[Peak]
            Initial peak guesses.
        x : ndarray
            X values.
        y : ndarray
            Y values.
        baseline : ndarray, optional
            Baseline values.

        Returns
        -------
        list[Peak]
            Initialized peak list (possibly with adjusted heights).
        """
        peaks0 = copy.deepcopy(peaks)

        if self.reheight:
            sig = y - (baseline if baseline is not None else 0.0)
            med = float(np.median(sig))
            for i, pk in enumerate(peaks0):
                if getattr(pk, "lock_height", False):
                    continue
                y_at = float(np.interp(pk.center, x, sig))
                peaks0[i] = replace(pk, height=max(y_at - med, 1e-6))

        return peaks0


class SolverDispatcher:
    """Dispatches fitting to appropriate solver."""

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize.

        Parameters
        ----------
        cfg : dict[str, Any]
            Configuration with solver selection and options.
        """
        self.cfg = cfg

    def prepare_options(self) -> Dict[str, Any]:
        """Prepare options for solver call.

        Returns
        -------
        dict[str, Any]
            Merged solver options.
        """
        solver_name = self.cfg.get("solver", "modern_vp")
        solver_opts = dict(self.cfg.get(solver_name, {}))
        opts = dict(solver_opts)
        opts.update(
            {
                "solver": solver_name,
                "loss": self.cfg.get("solver_loss", opts.get("loss", "linear")),
                "weights": self.cfg.get("solver_weight", opts.get("weights", "none")),
                "f_scale": self.cfg.get("solver_fscale", opts.get("f_scale", 1.0)),
                "maxfev": self.cfg.get("solver_maxfev", opts.get("maxfev", 20000)),
                "restarts": 1,
                "jitter_pct": 0.0,
            }
        )
        return opts

    def get_solver_name(self) -> str:
        """Get name of selected solver.

        Returns
        -------
        str
            Solver name.
        """
        return str(self.cfg.get("solver", "modern_vp")).lower()


class DataValidator:
    """Validates input data and fit mask."""

    @staticmethod
    def validate_fit_mask(
        x: np.ndarray,
        fit_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Validate and convert fit mask.

        Parameters
        ----------
        x : ndarray
            X data.
        fit_mask : ndarray, optional
            Proposed fit mask.

        Returns
        -------
        ndarray
            Valid boolean fit mask.

        Raises
        ------
        ValueError
            If fit_mask shape doesn't match x.
        """
        mask_arr = np.asarray(fit_mask, bool)
        if mask_arr.shape != x.shape:
            raise ValueError(
                f"fit_mask shape {mask_arr.shape} must match x shape {x.shape}"
            )
        return mask_arr

    @staticmethod
    def validate_baseline_shape(
        baseline: Optional[np.ndarray],
        x: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Validate and convert baseline.

        Parameters
        ----------
        baseline : ndarray, optional
            Proposed baseline.
        x : ndarray
            X data.

        Returns
        -------
        ndarray or None
            Valid baseline array.

        Raises
        ------
        ValueError
            If baseline shape doesn't match x.
        """
        if baseline is None:
            return None

        baseline_arr = np.asarray(baseline, float)
        if baseline_arr.shape != x.shape:
            raise ValueError("baseline shape mismatch")
        return baseline_arr
