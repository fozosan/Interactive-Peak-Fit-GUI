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
import time

import numpy as np
import pandas as pd

try:
    from scipy.optimize import nnls as _nnls
except Exception:  # pragma: no cover - optional dependency
    _nnls = None

from .fit_api import _vp_design_columns

__all__ = [
    "asymptotic_ci",
    "bootstrap_ci",
    "bayesian_ci",
    "UncertaintyResult",
    "finite_diff_jacobian",
]

log = logging.getLogger(__name__)

BOOT_BAND_MIN_SAMPLES = 16
BAYES_BAND_MIN_DRAWS = 50


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


def finite_diff_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Forward-difference Jacobian of ``residual_fn`` evaluated at ``theta``."""

    theta = np.asarray(theta, float)
    r0 = np.asarray(residual_fn(theta), float)
    m = r0.size
    p = theta.size
    J = np.empty((m, p), float)
    for j in range(p):
        h = eps * (1.0 + abs(theta[j]))
        th = theta.copy()
        th[j] += h
        r1 = np.asarray(residual_fn(th), float)
        J[:, j] = (r1 - r0) / h
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
    lo_q: float = 2.5,
    hi_q: float = 97.5,
    max_samples: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not theta_samples:
        raise ValueError("no theta samples")
    if len(theta_samples) > max_samples:
        sel = np.linspace(0, len(theta_samples) - 1, max_samples).astype(int)
        theta_samples = [theta_samples[i] for i in sel]
    preds = []
    for th in theta_samples:
        y = np.asarray(predict_full(np.asarray(th, float)), float).ravel()
        preds.append(y)
    Y = np.stack(preds, axis=0)
    lo = np.percentile(Y, lo_q, axis=0)
    hi = np.percentile(Y, hi_q, axis=0)
    return (np.asarray(x_all, float), lo, hi)


def _predict_full_vp(theta_nl: np.ndarray, fit_ctx: Dict[str, Any]) -> np.ndarray:
    x_fit = fit_ctx["x_fit"]
    y_fit = fit_ctx["y_target_fit"]
    w = fit_ctx.get("wvec_fit", None)
    x_all = fit_ctx["x_all"]
    base_all = fit_ctx["base_all"]
    add_mode = fit_ctx["add_mode"]
    vp_unpack_cw = fit_ctx["vp_unpack_cw"]
    etas = fit_ctx["vp_etas"]

    c_list, w_list = vp_unpack_cw(theta_nl)
    A_fit = _vp_design_columns(x_fit, c_list, w_list, etas)
    rhs = y_fit if w is None else (y_fit * w)
    Aeq = A_fit if w is None else (A_fit * w[:, None])

    if Aeq.shape[1] == 0:
        h = np.zeros(0, float)
    elif _nnls is not None:
        h, _ = _nnls(Aeq, rhs)
    else:
        h, *_ = np.linalg.lstsq(Aeq, rhs, rcond=None)
        h = np.clip(h, 0.0, np.inf)

    A_all = _vp_design_columns(x_all, c_list, w_list, etas)
    y_peaks = A_all @ h if h.size else np.zeros_like(x_all, float)
    return base_all + y_peaks if add_mode else y_peaks


def _prediction_band_vp(
    theta_nl_samples: Sequence[np.ndarray],
    fit_ctx: Dict[str, Any],
    lo_q: float = 2.5,
    hi_q: float = 97.5,
    max_samples: int = 256,
):
    if not theta_nl_samples:
        raise ValueError("no theta_nl samples")
    if len(theta_nl_samples) > max_samples:
        sel = np.linspace(0, len(theta_nl_samples) - 1, max_samples).astype(int)
        theta_nl_samples = [theta_nl_samples[i] for i in sel]
    preds = [
        np.asarray(_predict_full_vp(th, fit_ctx), float).ravel()
        for th in theta_nl_samples
    ]
    Y = np.stack(preds, axis=0)
    lo = np.percentile(Y, lo_q, axis=0)
    hi = np.percentile(Y, hi_q, axis=0)
    return (fit_ctx["x_all"], lo, hi)


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


def bootstrap_ci(
    theta: np.ndarray,
    residual: np.ndarray,
    jacobian: np.ndarray,
    *,
    predict_full=None,
    x_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    param_names: Optional[Sequence[str]] = None,
    locked_mask: Optional[np.ndarray] = None,
    fit_ctx: Optional[Dict[str, Any]] = None,
    n_boot: int = 1000,
    seed: Optional[int] = None,
    workers: int = 0,
    alpha: float = 0.05,
    center_residuals: bool = True,
    return_band: bool = False,
) -> UncertaintyResult:
    """Residual bootstrap with refitting through ``fit_ctx``."""

    t0 = float(time.time())
    theta = np.asarray(theta, float)
    r = np.asarray(residual, float)
    jacobian = np.asarray(jacobian, float)  # unused, kept for API parity
    if center_residuals:
        r = r - r.mean()

    if x_all is None or y_all is None:
        raise ValueError("x_all and y_all required for residual bootstrap")

    refit = None
    if fit_ctx:
        if callable(fit_ctx.get("refit")):
            refit = fit_ctx["refit"]
        elif hasattr(fit_ctx.get("solver_adapter", None), "refit"):
            refit = fit_ctx["solver_adapter"].refit
    if refit is None:
        from . import fit_api as _fit_api
        from .data_io import peaks_to_dicts

        peaks_obj = fit_ctx.get("peaks") if fit_ctx else None
        mode = (fit_ctx.get("mode") if fit_ctx else "add") or "add"
        baseline = fit_ctx.get("baseline") if fit_ctx else None
        solver = (fit_ctx.get("solver") if fit_ctx else None) or "classic"

        def refit(theta_init, locked_mask, bounds, x, y):
            cfg = {"solver": solver, "mode": mode, "peaks": peaks_to_dicts(peaks_obj)}
            res = _fit_api.run_fit_consistent(
                x,
                y,
                cfg,
                theta_init=theta_init,
                locked_mask=locked_mask,
                bounds=bounds,
                baseline=baseline,
            )
            if not res.get("fit_ok", False):
                raise RuntimeError("refit failed")
            return np.asarray(res["theta"], float)

    P = theta.size
    free_mask = np.ones(P, bool) if locked_mask is None else ~np.asarray(locked_mask, bool)

    def one_boot(i: int) -> Optional[np.ndarray]:
        local = np.random.default_rng(None if seed is None else seed + i)
        idx = local.integers(0, r.size, size=r.size)
        r_star = r[idx]
        y_hat = y_all - residual
        y_star = y_hat + r_star
        try:
            th_new = np.asarray(refit(theta, locked_mask, bounds, x_all, y_star), float)
            th_new[~free_mask] = theta[~free_mask]
            if bounds is not None:
                lo, hi = bounds
                if lo is not None:
                    th_new = np.maximum(th_new, lo)
                if hi is not None:
                    th_new = np.minimum(th_new, hi)
            return th_new
        except Exception:
            return None

    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")

    if workers and workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            thetas = list(ex.map(one_boot, range(n_boot)))
    else:
        thetas = [one_boot(i) for i in range(n_boot)]

    theta_succ = [t for t in thetas if t is not None]
    n_success = len(theta_succ)
    n_fail = n_boot - n_success
    if n_success < 2:
        raise RuntimeError("Insufficient successful bootstrap refits")

    T = np.vstack(theta_succ)
    mean = T.mean(axis=0)
    sd = T.std(axis=0, ddof=1)
    qlo = np.quantile(T, alpha / 2, axis=0)
    qhi = np.quantile(T, 1 - alpha / 2, axis=0)

    names = param_names or [f"p{i}" for i in range(P)]
    stats = {
        name: {
            "est": float(mean[i]),
            "sd": float(sd[i]),
            "p2.5": float(qlo[i]),
            "p97.5": float(qhi[i]),
        }
        for i, name in enumerate(names)
    }

    pct_at_bounds = (
        float(
            np.mean(
                (
                    (bounds[0] is not None)
                    and np.any(np.isclose(T, bounds[0], rtol=0, atol=0), axis=1)
                )
                |
                (
                    (bounds[1] is not None)
                    and np.any(np.isclose(T, bounds[1], rtol=0, atol=0), axis=1)
                )
            )
        )
        if bounds
        else 0.0
    )

    band = None
    band_reason = None
    if return_band:
        if predict_full is None or n_success < BOOT_BAND_MIN_SAMPLES:
            band_reason = "missing model or insufficient samples"
        else:
            Ys = np.vstack([predict_full(th) for th in theta_succ])
            lo = np.quantile(Ys, alpha / 2, axis=0)
            hi = np.quantile(Ys, 1 - alpha / 2, axis=0)
            band = (x_all, lo, hi)

    diag = {
        "B": int(n_boot),
        "n_boot": int(n_boot),
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "seed": seed,
        "pct_at_bounds": pct_at_bounds,
        "runtime_s": float(time.time() - t0),
        "band_source": "bootstrap-percentile" if band is not None else None,
        "band_reason": band_reason,
    }

    return UncertaintyResult(
        method="bootstrap",
        label="Bootstrap",
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

    residual_fn_kw = kwargs.pop("residual_fn", None)
    param_names_kw = kwargs.pop("param_names", None)
    locked_mask_kw = kwargs.pop("locked_mask", None)

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
            "residual_fn": residual_fn_kw,
            "bounds": bounds,
            "param_names": param_names_kw,
            "locked_mask": locked_mask_kw,
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

    if residual_fn is None:
        raise ValueError("residual_fn required")

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
    diagnostics: Dict[str, object] = {"n_draws": int(len(theta_samples))}

    if return_band and ctx.get("predict_full") is None:
        return_band = False
        diagnostics["band_disabled_no_model"] = True
        if isinstance(fit_ctx, dict):
            diag_fc = ctx.get("diagnostics", {})
            diag_fc["band_disabled_no_model"] = True
            fit_ctx["diagnostics"] = diag_fc

    try:
        fc = ctx
        if fc.get("solver_kind") == "vp":
            if return_band:
                struct = fc.get("vp_struct", [])
                theta_nl_samples: List[np.ndarray] = []
                for th in theta_samples:
                    th = np.asarray(th, float)
                    tnl: List[float] = []
                    for i, s in enumerate(struct):
                        if s.get("ic") is not None:
                            tnl.append(th[4 * i + 0])
                        if s.get("iw") is not None:
                            tnl.append(th[4 * i + 2])
                    theta_nl_samples.append(np.asarray(tnl, float))
                if len(theta_nl_samples) >= BAYES_BAND_MIN_DRAWS:
                    band = _prediction_band_vp(theta_nl_samples, fc)
                else:
                    reason = "insufficient_vp_draws"
        else:
            pred = fc.get("predict_full")
            x_all = fc.get("x_all")
            if return_band and pred is not None and x_all is not None and len(theta_samples) >= BAYES_BAND_MIN_DRAWS:
                xb, lob, hib = _prediction_band_from_thetas(theta_samples, pred, np.asarray(x_all, float))
                band = (xb, _smooth_envelope(lob), _smooth_envelope(hib))
            elif return_band:
                reason = "missing_predict_full_or_draws"
    except Exception as e:  # pragma: no cover
        reason = f"band_failed:{type(e).__name__}"

    if band is None and reason:
        diagnostics["band_reason"] = reason

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
    diag.update(diagnostics)
    label = "Bayesian (MCMC)"
    return UncertaintyResult(
        method="bayesian",
        label=label,
        stats=stats,
        diagnostics=diag,
        band=band,
    )
