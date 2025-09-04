from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import numpy as np

from .data_io import canonical_unc_label
from . import uncertainty as unc


def route_uncertainty(
    method: str,
    *,
    theta_hat: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian: Any,
    model_eval: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    fit_ctx: Optional[Dict[str, Any]] = None,
    x_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    workers: int = 0,
    seed: Optional[int] = None,
    n_boot: int = 100,
    bayes_steps: int = 8000,
    bayes_burn: int = 2000,
) -> Any:
    """Dispatch to asymptotic, bootstrap, or bayesian. No fallback."""
    canon = canonical_unc_label(method)
    s = canon.lower()

    # Materialize J or wrap callable for asymptotic
    if callable(jacobian):
        J_fn = jacobian
        J_mat = None
    else:
        J_mat = np.asarray(jacobian, float)
        J_fn = lambda _th: J_mat  # type: ignore

    if s.startswith("asymptotic"):
        return unc.asymptotic_ci(theta_hat, residual_fn, J_fn, model_eval)

    if s.startswith("bootstrap"):
        if J_mat is None:
            J_mat = np.asarray(J_fn(theta_hat), float)
        r0 = residual_fn(theta_hat)
        return unc.bootstrap_ci(
            theta=theta_hat,
            residual=r0,
            jacobian=J_mat,
            predict_full=model_eval,
            x_all=x_all,
            y_all=y_all,
            fit_ctx=fit_ctx or {},
            n_boot=int(n_boot),
            workers=int(workers),
            seed=seed,
        )

    if s.startswith("bayesian"):
        return unc.bayesian_ci(
            theta_hat=theta_hat,
            model=model_eval,
            predict_full=model_eval,
            x_all=fit_ctx.get("x_all") if fit_ctx else x_all,
            y_all=fit_ctx.get("y_all") if fit_ctx else y_all,
            residual_fn=residual_fn,
            fit_ctx=fit_ctx or {},
            n_steps=int(bayes_steps),
            n_burn=int(bayes_burn),
            seed=seed,
            return_band=True,
        )

    raise ValueError(f"Unknown uncertainty method: '{method}' -> '{canon}'")
