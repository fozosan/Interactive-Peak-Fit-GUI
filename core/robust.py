"""Backward compatibility wrappers for weight helpers."""

from .weights import noise_weights, robust_weights

# ``irls_weights`` was the previous public name for ``robust_weights``
irls_weights = robust_weights

__all__ = ["noise_weights", "robust_weights", "irls_weights"]
