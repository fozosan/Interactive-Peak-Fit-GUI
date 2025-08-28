"""Light-weight uncertainty estimators used in the tests.

The real project exposes a rather feature rich uncertainty module.  For the
kata we implement a very small subset that mimics the public surface of the
original functions.  The goal is API compatibility and deterministic behaviour
rather than ultimate statistical rigour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, List

import logging
import warnings

import numpy as np
import pandas as pd

__all__ = ["asymptotic_ci", "bootstrap_ci", "bayesian_ci", "UncertaintyResult"]

log = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Light weight container for uncertainty estimates.

    The class exposes the canonical fields ``method``, ``label``, ``stats``
    and ``diagnostics`` together with an optional prediction ``band``.  A
    backwards compatible dictionary-like API is provided so that older code
    accessing keys such as ``param_mean`` continues to operate.
    """

    method: str
    label: str
    stats: Dict[str, Any]
    diagnostics: Dict[str, Any]
    band: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    _legacy_cache: Dict[str, Any] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    @property
    def method_label(self) -> str:
        return self.label

    @property  # pragma: no cover - simple alias
    def param_stats(self) -> Dict[str, Any]:
        return self.stats

    @property  # pragma: no cover - simple alias
    def meta(self) -> Dict[str, Any]:
        return self.diagnostics

    # -- Backwards compatibility -------------------------------------------------
    def _legacy(self) -> Dict[str, Any]:
        if self._legacy_cache is not None:
            return self._legacy_cache

        params: Dict[str, Dict[str, float]] = {}
        names: list[str] = []
        means: list[float] = []
        stds: list[float] = []
        q05: list[float] = []
        q95: list[float] = []
        for name, st in self.stats.items():
            est = float(st.get("est", np.nan))
            sd = float(st.get("sd", np.nan))
            lo = st.get("p2.5")
            hi = st.get("p97.5")
            params[name] = {
                "mean": est,
                "std": sd,
                "q05": lo,
                "q50": est,
                "q95": hi,
            }
            names.append(name)
            means.append(est)
            stds.append(sd)
            q05.append(lo if lo is not None else np.nan)
            q95.append(hi if hi is not None else np.nan)

        band_dict = None
        if self.band is not None:
            x, lo, hi = self.band
            band_dict = {"x": x, "lo": lo, "hi": hi}

        diag = {
            "ess": self.diagnostics.get("ess"),
            "rhat": self.diagnostics.get("rhat"),
            "n_samples": self.diagnostics.get("n_samples"),
        }

        df = _param_df(
            np.asarray(means, float),
            np.asarray(stds, float),
            np.asarray(q05, float),
            np.asarray(q95, float),
            names,
        )

        self._legacy_cache = {
            "method": self.method,
            "params": params,
            "band": band_dict,
            "diagnostics": diag,
            "param_mean": np.asarray(means, float),
            "param_std": np.asarray(stds, float),
            "param_stats": df,
        }
        return self._legacy_cache

    def __getitem__(self, key: str) -> Any:  # dict-like access
        return self._legacy()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._legacy().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._legacy()



# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------

def complex_step_jacobian(f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, h: float = 1e-30) -> np.ndarray:
    theta = np.asarray(theta, float)
    n = theta.size
    J = np.empty((f(theta).size, n), float)
    for j in range(n):
        step = np.zeros_like(theta, dtype=complex)
        step[j] = 1j * h
        y = f(theta.astype(complex) + step)
        if not np.iscomplexobj(y):
            raise TypeError("complex-step not supported")
        J[:, j] = np.imag(y) / h
    return J


def finite_diff_jacobian(f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, scale: float = 1e-6) -> np.ndarray:
    theta = np.asarray(theta, float)
    n = theta.size
    f0 = f(theta)
    J = np.empty((f0.size, n), float)
    for j in range(n):
        step = scale * max(1.0, abs(theta[j]))
        tp = theta.copy(); tp[j] += step
        tm = theta.copy(); tm[j] -= step
        J[:, j] = (f(tp) - f(tm)) / (2.0 * step)
    return J


def _z_value(alpha: float) -> float:
    if alpha == 0.05:
        return 1.959963984540054
    try:  # pragma: no cover - optional dependency
        from scipy.stats import norm  # type: ignore
        return float(norm.ppf(1 - alpha / 2))
    except Exception:  # pragma: no cover
        return float(np.abs(np.quantile(np.random.standard_normal(200000), 1 - alpha / 2)))


def _param_df(theta: np.ndarray, std: np.ndarray, lo: np.ndarray, hi: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names), "value": theta, "std": std, "ci_lo": lo, "ci_hi": hi})


# ---------------------------------------------------------------------------
# Prediction band helper
# ---------------------------------------------------------------------------

def _prediction_band_from_thetas(
    theta_samples: Sequence[np.ndarray],
    predict_full: Callable[[np.ndarray], np.ndarray],
    x_all: np.ndarray,
    max_samples: int = 512,
    lo_q: float = 2.5,
    hi_q: float = 97.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction band from parameter samples.

    Parameters
    ----------
    theta_samples:
        Iterable of parameter vectors compatible with ``predict_full``.
    predict_full:
        Callable returning model predictions over ``x_all`` for a given
        parameter vector.
    x_all:
        Abscissa over which predictions are evaluated.
    max_samples:
        Maximum number of samples used for band construction.  Samples are
        thinned deterministically to ensure reproducibility.
    lo_q, hi_q:
        Percentile bounds defining the band (default 2.5/97.5).
    """

    theta_samples = list(theta_samples)
    if not theta_samples:
        raise ValueError("No theta samples provided.")

    if len(theta_samples) > max_samples:
        idx = np.linspace(0, len(theta_samples) - 1, max_samples).astype(int)
        theta_samples = [theta_samples[i] for i in idx]

    preds: List[np.ndarray] = []
    for th in theta_samples:
        y = np.asarray(predict_full(np.asarray(th, float)), float)
        preds.append(y)
    Y = np.stack(preds, axis=0)

    lo = np.percentile(Y, lo_q, axis=0)
    hi = np.percentile(Y, hi_q, axis=0)
    x_all = np.asarray(x_all, float)
    if lo.shape != x_all.shape or hi.shape != x_all.shape:
        raise ValueError(
            f"Band shape mismatch: lo {lo.shape}, hi {hi.shape}, x {x_all.shape}"
        )
    return x_all, lo, hi


def _coerce_draws_to_thetas(
    draws: Any,
    param_names: Optional[List[str]],
    theta_hat: Optional[np.ndarray],
) -> List[np.ndarray]:
    """Coerce various draw formats into a list of theta vectors."""

    if draws is None:
        return []

    if isinstance(draws, np.ndarray):
        if draws.ndim == 2:
            return [draws[i, :].astype(float, copy=False) for i in range(draws.shape[0])]
        if draws.ndim == 1:
            return [draws.astype(float, copy=False)]

    if isinstance(draws, (list, tuple)):
        out: List[np.ndarray] = []
        for row in draws:
            out.append(np.asarray(row, float).ravel())
        return out

    if isinstance(draws, dict) and param_names:
        first = next(iter(draws.values()))
        n = np.asarray(first).shape[0]
        out: List[np.ndarray] = []
        for i in range(n):
            vec = []
            for name in param_names:
                arr = np.asarray(draws[name]).ravel()
                vec.append(arr[i])
            out.append(np.asarray(vec, float))
        return out

    if theta_hat is not None:
        return [np.asarray(theta_hat, float)]
    return []


# ---------------------------------------------------------------------------
# Asymptotic confidence intervals
# ---------------------------------------------------------------------------

def asymptotic_ci(
    theta_hat: np.ndarray,
    residual: Any,
    jacobian: Any,
    ymodel_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 0.05,
    **_ignored: Any,
) -> UncertaintyResult:
    """Return asymptotic parameter statistics and a prediction band.

    ``residual`` and ``jacobian`` may be arrays already evaluated at
    ``theta_hat`` or callables accepting a parameter vector.  Extra keyword
    arguments are ignored for backwards compatibility.
    """

    theta = np.asarray(theta_hat, float)
    r = residual(theta) if callable(residual) else np.asarray(residual, float)
    J = jacobian(theta) if callable(jacobian) else np.asarray(jacobian, float)

    m, n = J.shape
    dof = max(m - n, 1)
    rss = float(np.dot(r, r))
    s2 = rss / dof

    JTJ = J.T @ J
    try:
        U, s, Vt = np.linalg.svd(JTJ, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        s = np.linalg.svd(JTJ, compute_uv=False)
        U = Vt = np.eye(JTJ.shape[0])
    cond = float(s[0] / s[-1]) if s[-1] != 0 else float("inf")
    if cond > 1e8:  # pragma: no cover - warning path
        warnings.warn("Ill conditioned Jacobian", RuntimeWarning)
    s_inv = np.array([1 / si if si > 1e-12 * s[0] else 0.0 for si in s])
    JTJ_inv = (Vt.T * (s_inv ** 2)) @ Vt
    cov = s2 * JTJ_inv

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = _z_value(alpha)
    ci_lo = theta - z * std
    ci_hi = theta + z * std

    y0 = np.asarray(ymodel_fn(theta), float)
    G = finite_diff_jacobian(ymodel_fn, theta)
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    band_std = np.sqrt(np.maximum(var, 0.0))
    lo = y0 - z * band_std
    hi = y0 + z * band_std
    x = np.arange(y0.size)

    names = [f"p{i}" for i in range(theta.size)]
    stats = {
        n: {
            "est": float(theta[i]),
            "sd": float(std[i]),
            "p2.5": float(ci_lo[i]),
            "p97.5": float(ci_hi[i]),
        }
        for i, n in enumerate(names)
    }

    band = (x, lo, hi)
    diag: Dict[str, object] = {"alpha": alpha, "param_order": names}
    label = "Asymptotic (JᵀJ)"
    return UncertaintyResult(
        method="asymptotic",
        label=label,
        stats=stats,
        diagnostics=diag,
        band=band,
    )


# ---------------------------------------------------------------------------
# Residual bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(*args: Any, fit_ctx: Optional[Dict[str, Any]] = None, **kwargs: Any) -> UncertaintyResult:
    """Residual bootstrap with extensive compatibility shim.

    ``fit_ctx`` may supply ``predict_full`` and ``x_all`` used for
    prediction band construction.  When omitted the information is extracted
    from the provided fit object if possible.
    """

    alias_map = {
        "n": "n_boot",
        "n_resamples": "n_boot",
        "seed_root": "seed",
        "random_state": "seed",
        "max_workers": "workers",
        "n_jobs": "workers",
        "return_bands": "return_band",
        "prediction_band": "return_band",
        "conf_alpha": "alpha",
        "band_alpha": "alpha",
    }
    for old, new in list(alias_map.items()):
        if old in kwargs and new not in kwargs:
            kwargs[new] = kwargs.pop(old)

    band_percentiles = kwargs.pop("band_percentiles", None)
    n_boot = int(kwargs.pop("n_boot", 300))
    seed = kwargs.pop("seed", None)
    workers = int(kwargs.pop("workers", 0))
    alpha = float(kwargs.pop("alpha", 0.05))
    return_band = bool(kwargs.pop("return_band", True))
    if band_percentiles is not None:
        try:
            lo_p, hi_p = band_percentiles
            alpha = 1.0 - (hi_p - lo_p) / 100.0
        except Exception:  # pragma: no cover
            pass

    # Determine call style -------------------------------------------------
    if args and isinstance(args[0], dict):
        fit = args[0]
    elif "fit" in kwargs:
        fit = kwargs.pop("fit")
    elif "engine" in kwargs and "data" in kwargs:
        engine = kwargs.pop("engine")
        data = kwargs.pop("data")
        fit = engine(**data, return_jacobian=True, return_predictors=True)
    else:
        theta = kwargs.pop("theta", kwargs.pop("theta_hat", None))
        residual = kwargs.pop("residual")
        jac = kwargs.pop("jacobian")
        predict_full = kwargs.pop("predict_full", kwargs.pop("predict_fn"))
        bounds = kwargs.pop("bounds", None)
        param_names = kwargs.pop("param_names", None)
        locked_mask = kwargs.pop("locked_mask", None)
        fit = {
            "theta": np.asarray(theta, float),
            "residual": residual(theta) if callable(residual) else np.asarray(residual, float),
            "jacobian": jac(theta) if callable(jac) else np.asarray(jac, float),
            "predict_full": predict_full,
            "bounds": bounds,
            "param_names": param_names,
            "locked_mask": locked_mask,
        }

    for k in list(kwargs.keys()):
        log.debug("bootstrap_ci ignoring argument %s", k)

    theta = np.asarray(fit["theta"], float)
    r = -np.asarray(fit["residual"], float)
    J = np.asarray(fit["jacobian"], float)
    predict_full = fit.get("predict_full")
    bounds = fit.get("bounds")
    param_names = fit.get("param_names") or [f"p{i}" for i in range(theta.size)]
    locked = np.asarray(fit.get("locked_mask"), bool)
    if locked.size != theta.size:
        locked = np.zeros(theta.size, bool)
    x_full = fit.get("x")

    if fit_ctx is None and isinstance(fit, dict) and "fit_ctx" in fit:
        fit_ctx = dict(fit["fit_ctx"])
    if fit_ctx is None:
        fit_ctx = {}
    if "predict_full" not in fit_ctx and predict_full is not None:
        fit_ctx["predict_full"] = predict_full
    if "x_all" not in fit_ctx and x_full is not None:
        fit_ctx["x_all"] = x_full
    if "theta_hat" not in fit_ctx:
        fit_ctx["theta_hat"] = theta
    if "param_names" not in fit_ctx:
        fit_ctx["param_names"] = list(param_names)

    J_pinv = np.linalg.pinv(J)
    rng = np.random.default_rng(seed)
    theta_samples: list[np.ndarray] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, r.size, r.size)
        delta = J_pinv @ r[idx]
        th = theta + delta
        th[locked] = theta[locked]
        if bounds is not None:
            lo_b, hi_b = bounds
            th = np.clip(th, lo_b, hi_b)
        theta_samples.append(th)

    samples_arr = np.vstack(theta_samples) if theta_samples else theta[None, :]
    mean = samples_arr.mean(axis=0)
    std = samples_arr.std(axis=0, ddof=1)
    ci_lo = np.quantile(samples_arr, alpha / 2, axis=0)
    ci_hi = np.quantile(samples_arr, 1 - alpha / 2, axis=0)
    mean[locked] = theta[locked]
    std[locked] = 0.0
    ci_lo[locked] = theta[locked]
    ci_hi[locked] = theta[locked]
    stats = {
        name: {
            "est": float(mean[i]),
            "sd": float(std[i]),
            "p2.5": float(ci_lo[i]),
            "p97.5": float(ci_hi[i]),
        }
        for i, name in enumerate(param_names)
    }

    band = None
    reason = None
    ctx = fit_ctx or {}
    try:
        if (
            return_band
            and "predict_full" in ctx
            and "x_all" in ctx
            and len(theta_samples) >= 8
        ):
            band = _prediction_band_from_thetas(
                theta_samples,
                ctx["predict_full"],
                np.asarray(ctx["x_all"], float),
                max_samples=512,
            )
        else:
            reason = "missing_predict_full_or_insufficient_samples"
    except Exception as e:  # pragma: no cover
        reason = f"band_failed:{type(e).__name__}"
        band = None

    if workers:  # pragma: no cover - workers ignored in this simplified impl
        log.debug("bootstrap_ci called with workers=%d (serial)", workers)

    diag: Dict[str, object] = {
        "B": int(n_boot),
        "n_success": int(len(theta_samples)),
    }
    if reason:
        diag["band_reason"] = reason
    label = "Bootstrap (residual)"
    return UncertaintyResult(
        method="bootstrap",
        label=label,
        stats=stats,
        diagnostics=diag,
        band=band,
    )


# ---------------------------------------------------------------------------
# Bayesian (emcee) uncertainty
# ---------------------------------------------------------------------------

def _smooth_envelope(y: np.ndarray) -> np.ndarray:
    if y.size < 5:
        return y
    try:  # pragma: no cover - optional dependency
        from scipy.signal import savgol_filter  # type: ignore

        win = 5 if y.size >= 5 else y.size | 1
        win = win if win % 2 == 1 else win + 1
        win = min(win, y.size if y.size % 2 == 1 else y.size - 1)
        return savgol_filter(y, win, 2, mode="interp")
    except Exception:
        k = min(5, y.size)
        kernel = np.ones(k) / k
        pad = k // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(yp, kernel, mode="valid")


def bayesian_ci(
    engine: Optional[Any] = None,
    data: Optional[dict] = None,
    model: Optional[Any] = None,
    theta_hat: Optional[np.ndarray] = None,
    bounds: Optional[Any] = None,
    seed: Optional[int] = None,
    n_walkers: int = 32,
    n_burn: int = 1000,
    n_steps: int = 2000,
    thin: int = 10,
    return_band: bool = False,
    fit_ctx: Optional[Dict[str, Any]] = None,
    x_all: Optional[np.ndarray] = None,
    base_all: Optional[np.ndarray] = None,
    add_mode: bool = False,
    **kwargs: Any,
) -> UncertaintyResult:
    """Bayesian posterior sampling using :mod:`emcee`.

    Parameters mirror the legacy implementation and extra ``kwargs`` are
    silently ignored after alias normalisation."""

    try:
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover - missing dependency
        raise ImportError("emcee is required for bayesian_ci") from exc

    alias_map = {
        "samples": "n_steps",
        "walkers": "n_walkers",
        "burn": "n_burn",
        "random_state": "seed",
        "prediction_band": "return_band",
    }
    for old, new in alias_map.items():
        if old in kwargs and new not in kwargs:
            locals()[new] = kwargs.pop(old)  # type: ignore[misc]

    # normalise recognised kwargs overriding defaults
    if "n_steps" in kwargs:
        n_steps = int(kwargs.pop("n_steps"))
    if "n_walkers" in kwargs:
        n_walkers = int(kwargs.pop("n_walkers"))
    if "n_burn" in kwargs:
        n_burn = int(kwargs.pop("n_burn"))
    if "seed" in kwargs:
        seed = kwargs.pop("seed")
    if "return_band" in kwargs:
        return_band = bool(kwargs.pop("return_band"))

    for k in list(kwargs.keys()):
        log.debug("bayesian_ci ignoring argument %s", k)
        kwargs.pop(k)

    # Determine call style -------------------------------------------------
    if isinstance(engine, dict) and data is None and model is None and theta_hat is None:
        fit = engine
    elif engine is not None and data is not None:
        fit = engine(**data, return_jacobian=True, return_predictors=True)
    else:
        fit = {
            "theta": np.asarray(theta_hat, float) if theta_hat is not None else None,
            "predict_full": model,
            "residual_fn": kwargs.get("residual_fn"),
            "bounds": bounds,
            "param_names": kwargs.get("param_names"),
            "locked_mask": kwargs.get("locked_mask"),
            "x": x_all,
            "baseline": base_all,
            "mode": "add" if add_mode else "subtract",
        }

    theta = np.asarray(fit.get("theta", theta_hat), float)
    residual_fn = fit.get("residual_fn")
    predict_full = fit.get("predict_full")
    bounds = fit.get("bounds", bounds)
    param_names = fit.get("param_names") or [f"p{i}" for i in range(theta.size)]
    locked = np.asarray(fit.get("locked_mask"), bool)
    if locked.size != theta.size:
        locked = np.zeros(theta.size, bool)
    x_all = fit.get("x", x_all)
    base_all = fit.get("baseline", base_all)
    add_mode = bool(fit.get("mode", "add") == "add")

    if fit_ctx is None and isinstance(fit, dict) and "fit_ctx" in fit:
        fit_ctx = dict(fit["fit_ctx"])
    if fit_ctx is None:
        fit_ctx = {}
    if "predict_full" not in fit_ctx and predict_full is not None:
        fit_ctx["predict_full"] = predict_full
    if "x_all" not in fit_ctx and x_all is not None:
        fit_ctx["x_all"] = x_all
    if "theta_hat" not in fit_ctx:
        fit_ctx["theta_hat"] = theta
    if "param_names" not in fit_ctx:
        fit_ctx["param_names"] = list(param_names)

    if residual_fn is None or predict_full is None:
        raise ValueError("residual_fn and predict_full required")

    r0 = residual_fn(theta)
    rss = float(np.dot(r0, r0))
    dof = max(r0.size - theta.size, 1)
    s2 = rss / dof

    def loglike(th: np.ndarray) -> float:
        if bounds is not None:
            lo_b, hi_b = bounds
            if np.any(th < lo_b) or np.any(th > hi_b):
                return -np.inf
        r = residual_fn(th)
        return -0.5 * np.dot(r, r) / s2

    free_idx = np.where(~locked)[0]
    ndim = int(free_idx.size)
    if ndim == 0:
        y0 = predict_full(theta)
        x = x_all if x_all is not None else np.arange(y0.size)
        band = (x, y0, y0) if return_band else None
        stats = {
            name: {"est": float(theta[i]), "sd": 0.0, "p2.5": float(theta[i]), "p97.5": float(theta[i])}
            for i, name in enumerate(param_names)
        }
        diag = {"n_samples": 0, "param_order": param_names}
        label = "Bayesian (MCMC)"
        return UncertaintyResult(
            method="bayesian",
            label=label,
            stats=stats,
            diagnostics=diag,
            band=band,
        )

    n_walkers = max(n_walkers, 2 * ndim)
    np.random.seed(seed)

    def log_prob_free(th_free: np.ndarray) -> float:
        th = theta.copy()
        th[free_idx] = th_free
        return loglike(th)

    p0 = theta[free_idx] + 1e-4 * np.random.standard_normal((n_walkers, ndim))
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_free)
    sampler.run_mcmc(p0, n_burn + n_steps, progress=False)
    chain = sampler.get_chain(discard=n_burn, thin=thin, flat=False)
    flat = chain.reshape(-1, ndim)

    samples_full = np.tile(theta, (flat.shape[0], 1))
    samples_full[:, free_idx] = flat

    mean = samples_full.mean(axis=0)
    std = samples_full.std(axis=0, ddof=1)
    ci_lo = np.quantile(samples_full, 0.025, axis=0)
    ci_hi = np.quantile(samples_full, 0.975, axis=0)

    mean[locked] = theta[locked]
    std[locked] = 0.0
    ci_lo[locked] = theta[locked]
    ci_hi[locked] = theta[locked]

    draws = samples_full
    ctx = fit_ctx or {}
    theta_samples = _coerce_draws_to_thetas(
        draws, ctx.get("param_names"), ctx.get("theta_hat")
    )
    band = None
    reason = None
    try:
        if (
            return_band
            and "predict_full" in ctx
            and "x_all" in ctx
            and len(theta_samples) >= 50
        ):
            xb, lob, hib = _prediction_band_from_thetas(
                theta_samples,
                ctx["predict_full"],
                np.asarray(ctx["x_all"], float),
                max_samples=512,
            )
            band = (xb, _smooth_envelope(lob), _smooth_envelope(hib))
        else:
            reason = "missing_predict_full_or_insufficient_samples"
    except Exception as e:  # pragma: no cover
        reason = f"band_failed:{type(e).__name__}"
        band = None

    # Diagnostics ---------------------------------------------------------
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        ess = flat.shape[0] / tau
    except Exception:  # pragma: no cover - autocorr failure
        ess = np.full(ndim, flat.shape[0])

    N = chain.shape[0]
    K = chain.shape[1]
    mean_chain = chain.mean(axis=0)
    var_chain = chain.var(axis=0, ddof=1)
    B = N * mean_chain.var(axis=0, ddof=1)
    W = var_chain.mean(axis=0)
    var_hat = ((N - 1) / N) * W + B / N
    rhat = np.sqrt(var_hat / W)
    if np.any(ess < 200) or np.any(rhat > 1.1):
        warnings.warn("MCMC diagnostics indicate poor convergence", RuntimeWarning)

    stats = {
        name: {
            "est": float(mean[i]),
            "sd": float(std[i]),
            "p2.5": float(ci_lo[i]),
            "p97.5": float(ci_hi[i]),
        }
        for i, name in enumerate(param_names)
    }

    diag: Dict[str, object] = {
        "ess": {name: float(ess[i]) for i, name in enumerate(param_names)},
        "rhat": {name: float(rhat[i]) for i, name in enumerate(param_names)},
        "n_samples": int(samples_full.shape[0]),
        "n_chains": int(chain.shape[1]),
        "param_order": param_names,
    }
    if reason:
        diag["band_reason"] = reason
    label = "Bayesian (MCMC)"
    return UncertaintyResult(
        method="bayesian",
        label=label,
        stats=stats,
        diagnostics=diag,
        band=band,
    )
