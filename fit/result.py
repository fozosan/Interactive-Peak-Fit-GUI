from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class FitResult:
    """Unified solver result.

    All solver backends are adapted to return this dataclass so that the rest of
    the application can interact with a stable API irrespective of which
    numerical routine performed the optimisation.
    """

    success: bool
    solver: str
    theta: np.ndarray
    peaks_out: List[Any]
    cost: float
    rmse: float
    nfev: int
    n_iter: int
    message: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    hit_bounds: bool = False
    hit_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
