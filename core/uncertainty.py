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
import math

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

try:
    from scipy.optimize import nnls as _nnls
except Exception:  # pragma: no cover - optional dependency
    _nnls = None

from .fit_api import _vp_design_columns
from .mcmc_utils import ess_autocorr, rhat_split

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
    # Optional abort + progress callback from fit_ctx
    abort_evt = None
    progress_cb = None
    if fit_ctx:
        abort_evt = fit_ctx.get("abort_event")
        progress_cb = fit_ctx.get("progress_cb")

    """Residual bootstrap with refitting through ``fit_ctx``.

    Notes:
      * Requires ``x_all`` and ``y_all`` for residual resampling.
      * Uses CPU parallelism (ThreadPool) when ``workers>0`` to evaluate prediction band.
      * If the model backend returns CuPy arrays, band aggregation converts to NumPy to avoid GPU saturation.
    """

    t0 = float(time.time())
    theta = np.asarray(theta, float)
    r = np.asarray(residual, float)
    jacobian = np.asarray(jacobian, float)  # unused, kept for API parity
    if center_residuals:
        r = r - r.mean()

    fit = fit_ctx or {}
    x_all = fit.get("x_all", x_all)
    y_all = fit.get("y_all", y_all)
    if x_all is None or y_all is None:
        raise ValueError("x_all and y_all required for residual bootstrap")
    x_all = np.asarray(x_all, float)
    y_all = np.asarray(y_all, float)

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
        # Provide progress + abort to the fallback path too
        _progress_cb = progress_cb
        _abort_evt = abort_evt
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
            th = np.asarray(res.get("theta", theta_init), float)
            ok = bool(res.get("fit_ok", False))
            return th, ok

    theta0 = np.asarray(fit.get("theta0", theta), float)
    P = theta.size

    # ---- Bootstrap resampling loop (abort-aware with progress pulses) ----
    # Existing variables used below (expected from prior code):
    #   r = centered residuals, theta0 = starting theta, locked_mask, bounds,
    #   refit() function, names/param_names, etc.
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")

    n = int(x_all.size)
    rng = np.random.default_rng(seed)
    T_list: List[np.ndarray] = []
    n_success = 0
    n_fail = 0
    aborted = False
    # Throttle progress to ~10–20 pulses over the run
    next_pulse_at = 0
    pulse_step = max(1, int(n_boot // 20))
    last_pulse_t = time.monotonic()
    for b in range(int(n_boot)):
        # Abort quickly if requested
        if abort_evt is not None and getattr(abort_evt, "is_set", None):
            try:
                if abort_evt.is_set():
                    aborted = True
                    break
            except Exception:
                pass
        # Progress pulse (time + count throttled)
        if progress_cb is not None:
            if b >= next_pulse_at or (time.monotonic() - last_pulse_t) > 0.5:
                try:
                    progress_cb(f"Bootstrap: {b}/{int(n_boot)}")
                except Exception:
                    pass
                last_pulse_t = time.monotonic()
                next_pulse_at = b + pulse_step

        # Residual resample
        idx = rng.integers(0, n, size=n)
        eps = r[idx]
        yb = y_all + eps
        try:
            ref_res = refit(theta0, locked_mask, bounds, x_all, yb)
            if isinstance(ref_res, tuple):
                th_b, ok = ref_res
            else:
                th_b, ok = ref_res, True
            if ok and np.all(np.isfinite(th_b)):
                T_list.append(th_b)
                n_success += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

    if len(T_list):
        theta_succ = np.vstack(T_list)
    else:
        theta_succ = np.empty((0, theta.size), float)
    T = theta_succ
    if n_success < 2:
        raise RuntimeError("Insufficient successful bootstrap refits")

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
    diagnostics: Dict[str, object] = {}
    if return_band:
        if predict_full is None or n_success < BOOT_BAND_MIN_SAMPLES:
            band_reason = "missing model or insufficient samples"
        else:
            max_use = min(n_success, 4096)
            sel = np.linspace(0, n_success - 1, max_use, dtype=int)

            def _eval_one(idx: int) -> np.ndarray:
                th = theta_succ[idx]
                yhat = predict_full(th)
                try:
                    import cupy as cp  # type: ignore
                    if isinstance(yhat, cp.ndarray):
                        return cp.asnumpy(yhat)
                except Exception:
                    pass
                return np.asarray(yhat)

            Y_list: List[np.ndarray] = []
            # Cap workers to CPU count
            w_req = int(workers)
            w = max(0, min(w_req, (os.cpu_count() or 1)))
            if w > 0:
                with ThreadPoolExecutor(max_workers=w) as ex:
                    futs = {ex.submit(_eval_one, int(i)): int(i) for i in sel}
                    for f in as_completed(futs):
                        Y_list.append(f.result())
            else:
                for i in sel:
                    Y_list.append(_eval_one(int(i)))

            Y = np.vstack(Y_list)
            lo = np.quantile(Y, alpha / 2, axis=0)
            hi = np.quantile(Y, 1 - alpha / 2, axis=0)
            band = (x_all, lo, hi)
            diagnostics["workers_used"] = int(w)
            diagnostics["band_backend"] = "numpy"
            # Free any CuPy cached blocks if available
            try:
                import cupy as cp  # type: ignore
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    diag = {
        "B": int(n_boot),
        "n_boot": int(n_boot),
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "seed": seed,
        "pct_at_bounds": pct_at_bounds,
        "aborted": bool(aborted),
        "runtime_s": float(time.time() - t0),
        "band_source": "bootstrap-percentile" if band is not None else None,
        "band_reason": band_reason,
    }
    diag.update(diagnostics)

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
    theta_hat: np.ndarray, *,
    model=None, predict_full=None, x_all=None, y_all=None,
    residual_fn=None, bounds=None, param_names=None, locked_mask=None,
    fit_ctx=None, n_walkers=None, n_burn=2000, n_steps=8000, thin=1,
    seed=None, return_band=True, prior_sigma="half_cauchy"
) -> UncertaintyResult:
    """
    Sample posterior of free θ and σ with emcee. Deterministic with seed.
    Returns posterior stats and optional posterior predictive band.
    """
    try:
        import emcee
    except Exception as e:
        raise RuntimeError("Bayesian method requires emcee>=3") from e

    theta_hat = np.asarray(theta_hat, float)
    locked = np.asarray(locked_mask, bool) if locked_mask is not None else np.zeros(theta_hat.size, bool)
    free_idx = np.where(~locked)[0]
    th_free = theta_hat[free_idx]
    P_free = th_free.size
    if P_free == 0:
        free_idx = np.array([], int)

    pred = predict_full or model
    if pred is None and residual_fn is not None:
        if x_all is None:
            x_all = np.arange(residual_fn(theta_hat).size, dtype=float)
        def pred_from_res(th):
            return (y_all if y_all is not None else 0.0) - residual_fn(th)
        pred = pred_from_res
    if pred is None or x_all is None or y_all is None:
        raise ValueError("predict_full/model and x_all, y_all are required for Bayesian")

    x_all = np.asarray(x_all, float)
    y_all = np.asarray(y_all, float)
    n = y_all.size

    lo = np.full(theta_hat.size, -np.inf)
    hi = np.full(theta_hat.size,  np.inf)
    if bounds is not None:
        lo_b, hi_b = bounds
        if lo_b is not None: lo = np.asarray(lo_b, float)
        if hi_b is not None: hi = np.asarray(hi_b, float)
    lo_f = lo[free_idx]; hi_f = hi[free_idx]

    resid0 = y_all - pred(theta_hat)
    rmse = float(np.sqrt(np.mean(resid0**2))) if resid0.size else 1.0
    if not math.isfinite(rmse) or rmse <= 0: rmse = 1.0

    th_loc = th_free.copy()
    th_scale = np.maximum(1e-6, np.where(np.isfinite(hi_f-lo_f), 0.1*(hi_f-lo_f), 10*np.maximum(1e-6, np.abs(th_free))))
    def log_prior(th_f, log_sigma):
        if np.any(th_f < lo_f) or np.any(th_f > hi_f):
            return -np.inf
        lp = -0.5*np.sum(((th_f - th_loc)/th_scale)**2)
        sigma = np.exp(log_sigma)
        if prior_sigma == "half_normal":
            lp += -0.5*(sigma/rmse)**2 + math.log(2.0) + math.log(1.0/max(sigma,1e-300))
        else:
            s = rmse
            lp += math.log(2.0/ math.pi) - math.log(s*(1.0 + (sigma/s)**2)) + math.log(1.0/max(sigma,1e-300))
        return lp

    def log_likelihood(th_f, log_sigma):
        th_full = theta_hat.copy()
        th_full[free_idx] = th_f
        mu = pred(th_full)
        if mu.shape != y_all.shape: return -np.inf
        sigma = np.exp(log_sigma)
        if not np.isfinite(sigma) or sigma <= 0: return -np.inf
        r = y_all - mu
        return -0.5*np.sum((r/sigma)**2) - n*np.log(sigma) - 0.5*n*math.log(2*math.pi)

    def log_prob(z):
        th_f = z[:-1]
        log_sigma = z[-1]
        lp = log_prior(th_f, log_sigma)
        if not np.isfinite(lp): return -np.inf
        return lp + log_likelihood(th_f, log_sigma)

    dim = P_free + 1
    if n_walkers is None:
        n_walkers = max(4*dim, 16)
    rng = np.random.default_rng(seed)
    # Optional progress + abort
    abort_evt = None
    progress_cb = None
    if fit_ctx:
        abort_evt, progress_cb = fit_ctx.get("abort_event"), fit_ctx.get("progress_cb")
    np.random.seed(seed)
    p0 = np.empty((n_walkers, dim), float)
    for k in range(n_walkers):
        jitter = rng.normal(scale=0.01, size=P_free)
        th0 = th_free + np.maximum(th_scale, 1e-3) * jitter
        th0 = np.clip(th0, lo_f, hi_f)
        p0[k, :P_free] = th0
        p0[k, -1] = math.log(max(rmse*abs(rng.normal(loc=1.0, scale=0.1)), 1e-6))

    # Optional parallel pool for emcee log_prob; cap workers
    workers_req = 0
    try:
        workers_req = int((fit_ctx or {}).get("unc_workers", 0))
    except Exception:
        workers_req = 0
    w = max(0, min(workers_req, (os.cpu_count() or 1)))
    pool = ThreadPoolExecutor(max_workers=w) if w > 0 else None
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers, dim, lambda z: log_prob(z), pool=pool
        )
        # Guardrails before sampling for very high dimension
        if dim > 40:
            n_burn = min(int(n_burn), 1000)
            n_steps = min(int(n_steps), 4000)

        draws = []
        accept = []
        aborted = False
        # Smaller chunks improve abort responsiveness
        chunk = 100 if (fit_ctx and fit_ctx.get("abort_event")) else 200
        state = p0
        total = n_burn + n_steps
        done = 0
        next_pulse_at = 0
        pulse_step = max(1, int(total // 20))
        last_pulse_t = time.monotonic()
        while done < total:
            step = min(chunk, total - done)
            state, lnp, _ = sampler.run_mcmc(state, step, progress=False, skip_initial_state_check=True)
            done += step
            # Progress pulse
            if progress_cb is not None:
                if done >= next_pulse_at or (time.monotonic() - last_pulse_t) > 0.5:
                    try:
                        progress_cb(f"Bayesian MCMC: {done}/{total}")
                    except Exception:
                        pass
                    last_pulse_t = time.monotonic()
                    next_pulse_at = done + pulse_step
            if abort_evt is not None and getattr(abort_evt, "is_set", None):
                try:
                    if abort_evt.is_set():
                        aborted = True
                        break
                except Exception:
                    pass
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    chain = sampler.get_chain(discard=n_burn, thin=thin)
    acc_frac = float(np.mean(sampler.acceptance_fraction))
    if chain.ndim != 3:
        chain = np.asarray(chain)
        chain = chain.reshape((n_walkers, -1, dim))
    n_samp = chain.shape[1]*n_walkers

    if n_samp < 2:
        raise RuntimeError("insufficient MCMC draws")
    flat = chain.reshape(-1, dim)
    th_draws = flat[:, :P_free]
    log_sigma_draws = flat[:, -1]
    sigma_draws = np.exp(log_sigma_draws)

    T = np.tile(theta_hat, (th_draws.shape[0], 1))
    if P_free:
        T[:, free_idx] = th_draws
    mean = T.mean(axis=0)
    sd = T.std(axis=0, ddof=1)
    qlo = np.quantile(T, 0.025, axis=0); qhi = np.quantile(T, 0.975, axis=0)

    names = param_names or [f"p{i}" for i in range(theta_hat.size)]
    stats = {names[i]: {"est": float(mean[i]), "sd": float(sd[i]), "p2.5": float(qlo[i]), "p97.5": float(qhi[i])}
             for i in range(theta_hat.size)}
    stats["sigma"] = {
        "est": float(np.mean(sigma_draws)),
        "sd": float(np.std(sigma_draws, ddof=1)),
        "p2.5": float(np.quantile(sigma_draws, 0.025)),
        "p97.5": float(np.quantile(sigma_draws, 0.975)),
    }

    # Skip emcee's get_autocorr_time() to avoid noisy warnings (short chains).
    chains = chain
    ess_min = ess_autocorr(chains)
    rhat_max = rhat_split(chains)
    diag = {
        "n_draws": int(n_samp),
        "ess_min": float(ess_min),
        "rhat_max": float(rhat_max),
        "accept_frac_mean": acc_frac,
        "seed": seed,
        "aborted": bool(aborted),
        "band_source": None,
    }

    band = None
    if return_band and predict_full is not None and x_all is not None and n_samp >= BAYES_BAND_MIN_DRAWS:
        max_use = min(n_samp, 4096)
        sel = np.linspace(0, n_samp-1, max_use, dtype=int)
        T_sel = T[sel]
        sig_sel = sigma_draws[sel]

        def _eval_one(idx: int) -> np.ndarray:
            t_i = T_sel[idx]
            s_i = float(sig_sel[idx])
            mu = pred(t_i)
            try:
                import cupy as cp  # type: ignore
                if isinstance(mu, cp.ndarray):
                    mu = cp.asnumpy(mu)
            except Exception:
                pass
            mu = np.asarray(mu, float)
            eps = rng.normal(0.0, s_i, size=mu.shape)
            return mu + eps

        Y_list: List[np.ndarray] = []
        # Thread predictive draws; cap workers
        workers_req = 0
        try:
            workers_req = int((fit_ctx or {}).get("unc_workers", 0))
        except Exception:
            workers_req = 0
        w = max(0, min(workers_req, (os.cpu_count() or 1)))
        if w > 0:
            with ThreadPoolExecutor(max_workers=w) as ex:
                futs = {ex.submit(_eval_one, int(i)): int(i) for i in range(len(sel))}
                for f in as_completed(futs):
                    Y_list.append(f.result())
        else:
            for i in range(len(sel)):
                Y_list.append(_eval_one(int(i)))

        Y = np.vstack(Y_list)
        lo = np.quantile(Y, 0.025, axis=0)
        hi = np.quantile(Y, 0.975, axis=0)
        band = (np.asarray(x_all, float), lo, hi)
        diag["band_source"] = "bayes-posterior-predictive"
        diag["band_backend"] = "numpy"
        diag["workers_used"] = int(w)
        try:
            import cupy as cp  # type: ignore
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return UncertaintyResult(
        method="bayesian",
        label="Bayesian (MCMC)",
        stats=stats,
        diagnostics=diag,
        band=band,
    )
