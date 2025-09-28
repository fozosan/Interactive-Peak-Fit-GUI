from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional
import numpy as np

from .data_io import canonical_unc_label
from . import uncertainty as unc


def _infer_alpha(ctx: Optional[Dict[str, Any]]) -> float:
    """Pull a CI alpha from a fit context if present; default 0.05 (95%)."""
    if not isinstance(ctx, dict):
        return 0.05
    for k in ("alpha", "unc_alpha", "ci_alpha"):
        try:
            if k in ctx and ctx[k] is not None:
                return float(ctx[k])
        except Exception:
            pass
    return 0.05


def _norm_jitter(val: Any) -> float:
    """
    Accept jitter as either a fraction (0..1) or percent (0..100).
    If > 1.5, treat as percent and divide by 100. Clamp to [0, 1].
    """
    try:
        f = float(val)
    except Exception:
        return 0.0
    if f < 0:
        f = 0.0
    if f > 1.5:
        f = f / 100.0
    if f > 1.0:
        f = 1.0
    return f


def _normalize_model_eval(
    model_eval: Optional[Callable[..., np.ndarray]],
    ctx: Dict[str, Any],
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Convert flexible predictors to the single-argument flavour our engines expect."""

    if not callable(model_eval):
        return None

    # Some legacy pathways still provide predict(theta, x) callables.  Bind x if we
    # can so the downstream uncertainty APIs see a predict(theta) interface.
    try:
        sig = inspect.signature(model_eval)
    except (TypeError, ValueError):  # pragma: no cover - very unusual callables
        return model_eval  # type: ignore[return-value]

    pos_args = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    if len(pos_args) >= 2:
        x_full = ctx.get("x_all")
        if x_full is None:
            return model_eval  # type: ignore[return-value]

        x_full = np.asarray(x_full, float)

        def _wrapped(theta: np.ndarray, _model=model_eval, _x=x_full):
            return _model(theta, _x)

        return _wrapped

    return model_eval  # type: ignore[return-value]


def route_uncertainty(
    method: str,
    *,
    theta_hat: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian: Any,
    model_eval: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fit_ctx: Optional[Dict[str, Any]] = None,
    x_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
    n_boot: int = 200,
) -> Any:
    """
    Route to the appropriate uncertainty engine and return an UncertaintyResult-like
    payload (dataclass from core.uncertainty or a mapping normalizable by data_io).

    Parameters
    ----------
    method
        User-facing label or alias; will be canonicalized.
    theta_hat
        Current parameter vector (length 4*N).
    residual_fn
        Callable that returns residuals for a given theta.
    jacobian
        Either a callable J(theta) or an already evaluated array.
    model_eval
        Callable yhat(theta) on the *fit window* for band construction (if supported).
    fit_ctx
        Optional context dict (x_all, y_all, baseline/mode, sharing flags, etc.).
    x_all, y_all
        Fit-window x and target y (if not carried inside fit_ctx).
    workers, seed
        Parallelism and RNG seed hints where applicable.
    n_boot
        Bootstrap draw count (only used if method resolves to Bootstrap and the caller
        chooses to route it here).
    """
    canon = canonical_unc_label(method)
    m = canon.lower()
    ctx = dict(fit_ctx or {})
    if x_all is not None:
        ctx.setdefault("x_all", x_all)
    if y_all is not None:
        ctx.setdefault("y_all", y_all)
    alpha = _infer_alpha(ctx)

    # Ensure we have a model evaluator on the fit window when bands are requested.
    # For Asymptotic and Bayesian, bands use predict_full/ymodel_fn(theta).
    if model_eval is None:
        # Fall back to any predictor the caller stashed in the context.
        maybe_pred = ctx.get("predict_full") or ctx.get("model")
        if callable(maybe_pred):
            model_eval = maybe_pred  # type: ignore[assignment]

    model_eval = _normalize_model_eval(model_eval, ctx)

    # --- ASYMPTOTIC ---------------------------------------------------------
    if "asymptotic" in m or "jᵀj" in m or "jtj" in m or "gauss" in m or "hessian" in m:
        if not callable(model_eval):
            # No predictor available — still return param stats without a band.
            ymodel = lambda th: residual_fn(th) * 0.0  # dummy; band will be ignored
        else:
            ymodel = model_eval
        return unc.asymptotic_ci(
            theta_hat=theta_hat,
            residual=residual_fn,
            jacobian=jacobian,
            ymodel_fn=ymodel,
            alpha=alpha,
        )

    # --- BOOTSTRAP ----------------------------------------------------------
    if "bootstrap" in m:
        # This path is *usually* handled directly in batch.runner for better control
        # over seeds and worker pools. We still support it here so GUI callers or
        # tests can route bootstrap through this function if they want.
        r0 = residual_fn(theta_hat)
        J = jacobian(theta_hat) if callable(jacobian) else np.asarray(jacobian, float)
        jitter = _norm_jitter(ctx.get("bootstrap_jitter", ctx.get("jitter", 0.0)))
        ctx["bootstrap_jitter"] = jitter
        return unc.bootstrap_ci(
            theta=theta_hat,
            residual=np.asarray(r0, float),
            jacobian=np.asarray(J, float),
            predict_full=model_eval,
            x_all=x_all if x_all is not None else ctx.get("x_all"),
            y_all=y_all if y_all is not None else ctx.get("y_all"),
            bounds=ctx.get("bounds"),
            param_names=ctx.get("param_names"),
            locked_mask=ctx.get("locked_mask"),
            fit_ctx=ctx,
            n_boot=int(n_boot),
            seed=seed,
            workers=workers if workers not in (False,) else None,
            alpha=alpha,
            center_residuals=bool(ctx.get("unc_center_resid", True)),
            jitter=jitter,
            return_band=True,
        )

    # --- BAYESIAN -----------------------------------------------------------
    if "bayes" in m or "mcmc" in m:
        # Pull common MCMC settings (provide safe fallbacks).
        n_walkers = ctx.get("bayes_walkers", None)
        n_burn = int(ctx.get("bayes_burn", 2000))
        n_steps = int(ctx.get("bayes_steps", 8000))
        thin = int(ctx.get("bayes_thin", 1))
        prior_sigma = str(ctx.get("bayes_prior_sigma", "half_cauchy"))

        return unc.bayesian_ci(
            theta_hat=theta_hat,
            model=model_eval,
            predict_full=model_eval,
            x_all=x_all if x_all is not None else ctx.get("x_all"),
            y_all=y_all if y_all is not None else ctx.get("y_all"),
            residual_fn=residual_fn,
            bounds=ctx.get("bounds"),
            param_names=ctx.get("param_names"),
            locked_mask=ctx.get("locked_mask"),
            fit_ctx=ctx,
            n_walkers=n_walkers,
            n_burn=n_burn,
            n_steps=n_steps,
            thin=thin,
            seed=seed,
            workers=workers,
            return_band=True,
            prior_sigma=prior_sigma,
        )

    raise ValueError(f"Unknown uncertainty method: '{method}' -> '{canon}'")
