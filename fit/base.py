"""Abstract base class for peak fitting solvers.

This module defines the common interface for all solver implementations,
enabling polymorphic solver selection and consistent behavior across
different optimization backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, TypedDict, Any
import numpy as np

from core.peaks import Peak


class SolveResult(TypedDict):
    """Standard result dictionary returned by all solvers.

    Keys
    ----
    success : bool
        Whether optimization converged.
    theta : ndarray
        Fitted parameter vector (full, all parameters).
    message : str
        Status message from solver.
    cost : float
        Final cost value (sum of squared residuals).
    jac : ndarray, optional
        Jacobian matrix at solution.
    cov : ndarray, optional
        Covariance/Hessian matrix at solution.
    peaks_out : list[Peak]
        Fitted Peak objects.
    y_fit : ndarray
        Fitted y values.
    meta : dict, optional
        Solver-specific metadata (nfev, nit, etc.).
    """

    success: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    peaks_out: List[Peak]
    y_fit: np.ndarray
    meta: Dict[str, Any]


class PeakFitSolver(ABC):
    """Abstract base class for peak fitting solvers.

    All solver implementations (Classic, Modern, Modern VP, LMFit, etc.)
    should inherit from this class and implement the required methods.
    This ensures consistent behavior and enables runtime polymorphism.
    """

    @abstractmethod
    def solve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        peaks: Sequence[Peak],
        mode: str,
        baseline: Optional[np.ndarray],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fit peaks to spectral data.

        Parameters
        ----------
        x : ndarray
            X values (wavelength, frequency, etc.).
        y : ndarray
            Y values (intensity).
        peaks : sequence[Peak]
            Initial peak guesses.
        mode : str
            Fitting mode: "add" (baseline + peaks) or "subtract" (peaks only).
        baseline : ndarray, optional
            Baseline values. Required for "add" mode.
        options : dict[str, Any]
            Solver-specific options (maxfev, loss, regularization, etc.).

        Returns
        -------
        dict[str, Any]
            SolveResult-compliant dictionary with keys:
            success, theta, message, cost, jac, cov, peaks_out, y_fit, meta.
        """
        pass

    def validate_inputs(
        self,
        x: np.ndarray,
        y: np.ndarray,
        peaks: Sequence[Peak],
        mode: str,
        baseline: Optional[np.ndarray],
    ) -> None:
        """Validate inputs before solving.

        Parameters
        ----------
        x, y : ndarray
            Data arrays.
        peaks : sequence[Peak]
            Peak guesses.
        mode : str
            Fitting mode.
        baseline : ndarray, optional
            Baseline.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if len(x) != len(y):
            raise ValueError(f"x and y lengths don't match: {len(x)} vs {len(y)}")
        if len(x) == 0:
            raise ValueError("x and y are empty")
        if mode not in ("add", "subtract"):
            raise ValueError(f"Invalid mode '{mode}', must be 'add' or 'subtract'")
        if mode == "add" and baseline is None:
            raise ValueError("baseline is required for 'add' mode")
        if baseline is not None and len(baseline) != len(x):
            raise ValueError(f"baseline length {len(baseline)} != x length {len(x)}")

    def check_result(
        self,
        result: Dict[str, Any],
        theta_expected_size: int,
    ) -> None:
        """Validate solver result before returning.

        Parameters
        ----------
        result : dict
            Result dictionary from solve().
        theta_expected_size : int
            Expected size of theta vector.

        Raises
        ------
        ValueError
            If result is invalid.
        """
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        required_keys = {"success", "theta", "message", "cost"}
        missing = required_keys - set(result.keys())
        if missing:
            raise ValueError(f"Result missing keys: {missing}")
        if not isinstance(result["success"], bool):
            raise ValueError("success must be boolean")
        theta = np.asarray(result["theta"])
        if len(theta.ravel()) != theta_expected_size:
            raise ValueError(
                f"theta size {len(theta.ravel())} != "
                f"expected {theta_expected_size}"
            )
        if not isinstance(result["cost"], (int, float)):
            raise ValueError("cost must be numeric")


class SolverRegistry:
    """Registry for managing solver implementations.

    This enables runtime solver selection and polymorphic behavior without
    hardcoded imports or conditionals.
    """

    _solvers: Dict[str, type[PeakFitSolver]] = {}

    @classmethod
    def register(cls, name: str, solver_class: type[PeakFitSolver]) -> None:
        """Register a solver implementation.

        Parameters
        ----------
        name : str
            Unique solver name (e.g., "classic", "modern", "modern_vp").
        solver_class : type[PeakFitSolver]
            Solver class (must implement PeakFitSolver interface).
        """
        if not issubclass(solver_class, PeakFitSolver):
            raise TypeError(f"{solver_class} must inherit from PeakFitSolver")
        cls._solvers[str(name).lower()] = solver_class

    @classmethod
    def get(cls, name: str) -> type[PeakFitSolver]:
        """Get solver by name.

        Parameters
        ----------
        name : str
            Solver name.

        Returns
        -------
        type[PeakFitSolver]
            Solver class.

        Raises
        ------
        KeyError
            If solver not registered.
        """
        name_lower = str(name).lower()
        if name_lower not in cls._solvers:
            available = ", ".join(sorted(cls._solvers.keys()))
            raise KeyError(
                f"Solver '{name}' not found. Available: {available}"
            )
        return cls._solvers[name_lower]

    @classmethod
    def available(cls) -> List[str]:
        """List all registered solver names.

        Returns
        -------
        list[str]
            Sorted list of available solver names.
        """
        return sorted(cls._solvers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered solvers (mainly for testing)."""
        cls._solvers.clear()


class AdapterSolver(PeakFitSolver):
    """Adapter to wrap existing solver functions as PeakFitSolver instances.

    This class enables incremental migration to the new ABC without
    refactoring all solver implementations at once.
    """

    def __init__(
        self,
        solve_func: callable,
        name: str = "adapter",
    ):
        """Initialize adapter.

        Parameters
        ----------
        solve_func : callable
            Function with signature solve(x, y, peaks, mode, baseline, options)
            returning a result dict.
        name : str
            Solver name for identification.
        """
        self._solve_func = solve_func
        self._name = name

    def solve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        peaks: Sequence[Peak],
        mode: str,
        baseline: Optional[np.ndarray],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call wrapped solver function."""
        self.validate_inputs(x, y, peaks, mode, baseline)
        result = self._solve_func(x, y, peaks, mode, baseline, options)
        theta_size = len(peaks) * 4  # 4 params per peak
        self.check_result(result, theta_size)
        return result
