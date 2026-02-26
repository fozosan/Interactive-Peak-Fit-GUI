"""Utilities for parameter packing/unpacking and structural handling.

This module consolidates parameter vector construction and extraction logic
that was previously scattered across fit_api.py, modern_vp.py, and other
modules. It provides a unified interface for working with parameter vectors
in the context of pseudo-Voigt peak fitting.
"""

from __future__ import annotations

from typing import List, Tuple, NamedTuple
import numpy as np

from .peaks import Peak


class SolverVectors(NamedTuple):
    """Vectors for nonlinear (c, w) parameters in variable-projection solver.

    Attributes
    ----------
    theta : ndarray
        Flattened vector of nonlinear parameters (centers and widths for
        unlocked peaks).
    bounds : tuple[ndarray, ndarray]
        Lower and upper bounds for theta.
    x_scale : ndarray
        Scaling vector for parameter normalization.
    indices : ndarray
        Original indices in theta_full for each parameter in theta.
    mapping : list[tuple[int, str]]
        List of (peak_index, kind) where kind is "c" (center) or "f" (fwhm).
    """

    theta: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    x_scale: np.ndarray
    indices: np.ndarray
    mapping: List[Tuple[int, str]]


def vp_unpack_centers_widths(
    theta_nl: np.ndarray,
    struct: List[dict],
    peaks: List[Peak],
) -> Tuple[List[float], List[float]]:
    """Unpack center and width parameters from nonlinear vector.

    This function consolidates the logic for reconstructing center (c) and
    width (w) parameters from the nonlinear solution vector, accounting for
    locked parameters and fixed values.

    Parameters
    ----------
    theta_nl : ndarray
        Vector of nonlinear parameters (centers and widths for unlocked peaks).
    struct : list[dict]
        Structure metadata for each peak, containing "ic", "iw" (indices)
        and "c_fixed", "w_fixed" (fixed values for locked parameters).
    peaks : list[Peak]
        Original Peak objects.

    Returns
    -------
    c_list : list[float]
        Center values for all peaks.
    w_list : list[float]
        Width (FWHM) values for all peaks.
    """
    theta_nl = list(np.asarray(theta_nl, float).ravel())
    c_list, w_list = [], []
    tptr = 0

    for s, pk in zip(struct, peaks):
        # Unpack center
        if s.get("ic") is not None:
            c = float(theta_nl[tptr])
            tptr += 1
        else:
            c = float(s.get("c_fixed", pk.center))

        # Unpack width
        if s.get("iw") is not None:
            w = float(theta_nl[tptr])
            tptr += 1
        else:
            w = float(s.get("w_fixed", pk.fwhm))

        c_list.append(c)
        w_list.append(w)

    return c_list, w_list


def create_parameter_unpacker(
    centers_init: np.ndarray,
    widths_init: np.ndarray,
    mapping: List[Tuple[int, str]],
) -> callable:
    """Create a function to unpack (c, w) from a reduced parameter vector.

    This factory creates a reusable unpacker that reconstructs full center
    and width arrays from a reduced vector of nonlinear parameters.

    Parameters
    ----------
    centers_init : ndarray
        Initial center values for all peaks.
    widths_init : ndarray
        Initial width values for all peaks.
    mapping : list[tuple[int, str]]
        List of (peak_index, kind) describing the parameter vector structure.
        kind is "c" (center) or "f" (width/fwhm).

    Returns
    -------
    unpack : callable
        Function with signature unpack(t: ndarray) -> tuple[ndarray, ndarray]
        that takes a reduced parameter vector and returns (centers, widths).
    """
    c0 = np.asarray(centers_init, float)
    f0 = np.asarray(widths_init, float)

    def unpack(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack centers and widths from reduced vector."""
        c = c0.copy()
        f = f0.copy()
        j = 0
        for pk_idx, kind in mapping:
            if kind == "c":
                c[pk_idx] = t[j]
            else:  # kind == "f"
                f[pk_idx] = t[j]
            j += 1
        return c, f

    return unpack


def pack_nonlinear_params(
    centers: np.ndarray,
    widths: np.ndarray,
    mapping: List[Tuple[int, str]],
) -> np.ndarray:
    """Pack (c, w) values into a reduced parameter vector.

    Inverse operation of the unpacker created by create_parameter_unpacker.

    Parameters
    ----------
    centers : ndarray
        Center values for all peaks.
    widths : ndarray
        Width values for all peaks.
    mapping : list[tuple[int, str]]
        List of (peak_index, kind) describing the parameter vector structure.

    Returns
    -------
    theta : ndarray
        Reduced parameter vector containing only the parameters specified
        in the mapping.
    """
    theta = []
    for pk_idx, kind in mapping:
        if kind == "c":
            theta.append(centers[pk_idx])
        else:  # kind == "f"
            theta.append(widths[pk_idx])
    return np.asarray(theta, float)


def validate_param_shapes(
    theta_full: np.ndarray,
    peaks: List[Peak],
    expected_params_per_peak: int = 4,
) -> None:
    """Validate that theta vector matches peak count and parameter structure.

    Parameters
    ----------
    theta_full : ndarray
        Full parameter vector (flattened).
    peaks : list[Peak]
        Peak objects to validate against.
    expected_params_per_peak : int, default=4
        Number of parameters per peak (typically 4 for pseudo-Voigt:
        height, center, fwhm, eta).

    Raises
    ------
    ValueError
        If theta_full size doesn't match expected size.
    """
    expected_size = len(peaks) * expected_params_per_peak
    actual_size = len(np.asarray(theta_full).ravel())
    if actual_size != expected_size:
        raise ValueError(
            f"Parameter vector size {actual_size} does not match "
            f"expected size {expected_size} for {len(peaks)} peaks with "
            f"{expected_params_per_peak} params each"
        )
