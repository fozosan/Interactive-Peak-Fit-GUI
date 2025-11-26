"""Utility helpers for uncertainty configuration mapping and validation."""
from __future__ import annotations

from typing import Any, Dict, Mapping

_DISPLAY_TO_KEY: Dict[str, str] = {
    "asymptotic (jᵀj)": "asymptotic",
    "bootstrap (percentile)": "bootstrap",
    "bayesian (mcmc)": "bayesian",
}


def method_display_to_key(label: str | None) -> str:
    """Map a GUI display label to a canonical configuration key.

    Parameters
    ----------
    label:
        Display label from the GUI combo box. Falls back to "asymptotic" when
        the label is missing or empty. Strings are normalized case-insensitively
        and tolerate prefix matches (e.g. "Bootstrap" → "bootstrap").
    """

    if label is None:
        return "asymptotic"

    normalized = str(label).strip()
    if not normalized:
        return "asymptotic"

    lower = normalized.lower()
    mapped = _DISPLAY_TO_KEY.get(lower)
    if mapped:
        return mapped

    if lower.startswith("asymptotic"):
        return "asymptotic"
    if lower.startswith("bootstrap"):
        return "bootstrap"
    if lower.startswith("bayes"):
        return "bayesian"

    # Already canonical or unknown label; pass through for forward compatibility.
    return lower


def normalize_float(value: Any, fallback: float) -> float:
    """Return ``value`` as ``float`` or ``fallback`` when conversion fails."""

    try:
        if value is None:
            raise ValueError
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            raise ValueError
        return float(text)
    except (TypeError, ValueError):
        return float(fallback)


def normalize_int(value: Any, fallback: int) -> int:
    """Return ``value`` as ``int`` or ``fallback`` when conversion fails."""

    try:
        if value is None:
            raise ValueError
        if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
            raise ValueError
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        text = str(value).strip()
        if not text:
            raise ValueError
        return int(float(text))
    except (TypeError, ValueError):
        return int(fallback)


def validate_unc_knobs(knobs: Mapping[str, Any]) -> Dict[str, int]:
    """Validate Bayesian uncertainty knobs from the GUI/configuration.

    Parameters
    ----------
    knobs:
        Mapping containing ``walkers``, ``burn``, ``steps`` and ``thin`` keys.

    Returns
    -------
    Dict[str, int]
        Normalized knob values suitable for downstream consumers.

    Raises
    ------
    ValueError
        If the supplied configuration is incomplete or inconsistent.
    """

    walkers = max(0, normalize_int(knobs.get("walkers"), 0))
    burn = max(0, normalize_int(knobs.get("burn"), 0))
    steps = normalize_int(knobs.get("steps"), 0)
    thin = normalize_int(knobs.get("thin"), 1)

    if steps <= 0:
        raise ValueError("Bayesian steps must be a positive integer")
    if thin <= 0:
        raise ValueError("Bayesian thinning must be ≥ 1")

    return {
        "walkers": walkers,
        "burn": burn,
        "steps": steps,
        "thin": thin,
    }


__all__ = [
    "method_display_to_key",
    "normalize_float",
    "normalize_int",
    "validate_unc_knobs",
]
