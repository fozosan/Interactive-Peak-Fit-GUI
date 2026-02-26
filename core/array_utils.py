"""Utilities for consistent array handling and shape validation.

This module consolidates common array manipulation patterns that were
previously scattered throughout the codebase, particularly `.reshape(-1)`
and array shape standardization operations.
"""

from __future__ import annotations

from typing import Union, Tuple
import numpy as np


def ensure_1d_array(
    arr: Union[np.ndarray, list, float],
    dtype: type = float,
    name: str = "array",
) -> np.ndarray:
    """Convert input to a 1D NumPy array with type coercion.

    Parameters
    ----------
    arr : ndarray, list, or scalar
        Input to convert.
    dtype : type, default=float
        Target dtype.
    name : str
        Name for error messages.

    Returns
    -------
    ndarray
        1D array of specified dtype.

    Raises
    ------
    ValueError
        If input cannot be converted to 1D array.
    """
    arr_np = np.asarray(arr, dtype=dtype)
    arr_1d = arr_np.ravel()
    if arr_1d.size == 0:
        raise ValueError(f"{name} is empty")
    return arr_1d


def ensure_vector(
    arr: Union[np.ndarray, list],
    dtype: type = float,
    name: str = "vector",
) -> np.ndarray:
    """Ensure input is a flat vector array.

    Equivalent to `.reshape(-1)` followed by dtype conversion.

    Parameters
    ----------
    arr : ndarray or list
        Input array.
    dtype : type, default=float
        Target dtype.
    name : str
        Name for error messages.

    Returns
    -------
    ndarray
        Flattened 1D array.
    """
    return np.asarray(arr, dtype=dtype).ravel()


def match_shapes(
    arrays: dict[str, np.ndarray],
    expected_shape: Tuple[int, ...],
) -> None:
    """Validate that all arrays match expected shape.

    Parameters
    ----------
    arrays : dict[str, ndarray]
        Dictionary mapping names to arrays.
    expected_shape : tuple[int, ...]
        Expected shape.

    Raises
    ------
    ValueError
        If any array doesn't match expected shape.
    """
    for name, arr in arrays.items():
        arr_np = np.asarray(arr)
        if arr_np.shape != expected_shape:
            raise ValueError(
                f"Array '{name}' shape {arr_np.shape} does not match "
                f"expected shape {expected_shape}"
            )


def broadcast_to_shape(
    arr: Union[np.ndarray, float],
    target_shape: Tuple[int, ...],
    name: str = "array",
) -> np.ndarray:
    """Broadcast array to target shape with shape validation.

    Parameters
    ----------
    arr : ndarray or scalar
        Array to broadcast.
    target_shape : tuple[int, ...]
        Target shape.
    name : str
        Name for error messages.

    Returns
    -------
    ndarray
        Broadcasted array.

    Raises
    ------
    ValueError
        If broadcast is not possible.
    """
    try:
        return np.broadcast_to(arr, target_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast '{name}' with shape {np.asarray(arr).shape} "
            f"to {target_shape}: {e}"
        )


def stack_columns(
    columns: list[np.ndarray],
    axis: int = 1,
) -> np.ndarray:
    """Stack 1D arrays as columns with shape validation.

    Equivalent to `np.column_stack()` but with explicit error handling.

    Parameters
    ----------
    columns : list[ndarray]
        List of 1D arrays to stack.
    axis : int, default=1
        Axis along which to stack (1 for columns, 0 for rows).

    Returns
    -------
    ndarray
        Stacked 2D array, or 1D if only one column.

    Raises
    ------
    ValueError
        If columns have mismatched lengths.
    """
    if not columns:
        raise ValueError("No columns to stack")

    if len(columns) == 1:
        return np.asarray(columns[0], dtype=float).ravel()

    cols_arr = [np.asarray(c, dtype=float).ravel() for c in columns]
    lengths = [len(c) for c in cols_arr]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"Column lengths {lengths} are not uniform. "
            f"All columns must have the same length."
        )

    if axis == 1:
        return np.column_stack(cols_arr)
    elif axis == 0:
        return np.row_stack(cols_arr)
    else:
        raise ValueError(f"Invalid axis {axis}, must be 0 or 1")


def safe_reshape(
    arr: np.ndarray,
    new_shape: Union[int, Tuple[int, ...]],
    name: str = "array",
) -> np.ndarray:
    """Reshape array with explicit error handling.

    Parameters
    ----------
    arr : ndarray
        Array to reshape.
    new_shape : int or tuple[int, ...]
        Target shape.
    name : str
        Name for error messages.

    Returns
    -------
    ndarray
        Reshaped array.

    Raises
    ------
    ValueError
        If reshape is not possible.
    """
    arr_np = np.asarray(arr)
    try:
        return arr_np.reshape(new_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot reshape '{name}' from shape {arr_np.shape} "
            f"to {new_shape}: {e}"
        )


def atleast_2d_column(
    arr: Union[np.ndarray, list, float],
    dtype: type = float,
) -> np.ndarray:
    """Convert to 2D column vector.

    Parameters
    ----------
    arr : ndarray, list, or scalar
        Input to convert.
    dtype : type, default=float
        Target dtype.

    Returns
    -------
    ndarray
        2D array with shape (n, 1).
    """
    arr_1d = ensure_vector(arr, dtype=dtype)
    return arr_1d.reshape(-1, 1)


def extract_diagonal_plus(
    arr: np.ndarray,
    name: str = "covariance matrix",
) -> np.ndarray:
    """Extract diagonal with validation for square matrices.

    Parameters
    ----------
    arr : ndarray
        2D array (typically covariance matrix).
    name : str
        Name for error messages.

    Returns
    -------
    ndarray
        Diagonal elements.

    Raises
    ------
    ValueError
        If input is not 2D or not square.
    """
    arr_np = np.asarray(arr)
    if arr_np.ndim != 2:
        raise ValueError(
            f"{name} must be 2D, got shape {arr_np.shape}"
        )
    if arr_np.shape[0] != arr_np.shape[1]:
        raise ValueError(
            f"{name} must be square, got shape {arr_np.shape}"
        )
    return np.diag(arr_np)
