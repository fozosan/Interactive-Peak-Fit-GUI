"""Utility helpers for uncertainty configuration mapping."""
from __future__ import annotations

from typing import Dict

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


__all__ = ["method_display_to_key"]
