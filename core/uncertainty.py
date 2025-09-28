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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

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

        def _to_float(x: Any) -> float:
            if x is None:
                return float("nan")
            try:
                return float(x)
            except Exception:
                return float("nan")

        def _seq_len(val: Any) -> int:
            if isinstance(val, np.ndarray):
                return int(val.size) if val.ndim > 0 else 0
            if isinstance(val, (list, tuple)):
                return len(val)
            return 0

        def _values_for(st: Dict[str, Any], key: str, n: int) -> list[float]:
            val = st.get(key)
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    seq = [val.item()] * n
                else:
                    seq = val.reshape(-1).tolist()
            elif isinstance(val, (list, tuple)):
                seq = list(val)
            elif val is None:
                seq = []
            else:
                seq = [val] * n
            out: list[float] = []
            for i in range(n):
                out.append(_to_float(seq[i] if i < len(seq) else float("nan")))
            return out

        block_keys = ("center", "height", "fwhm", "eta")
        max_len = 0
        for key in block_keys:
            st = self.stats.get(key)
            if not isinstance(st, dict):
                continue
            max_len = max(
                max_len,
                _seq_len(st.get("est")),
                _seq_len(st.get("sd")),
                _seq_len(st.get("p2_5")),
                _seq_len(st.get("p97_5")),
            )

        processed: set[str] = set()
        if max_len > 0:
            for key in block_keys:
                st = self.stats.get(key)
                if not isinstance(st, dict):
                    continue
                processed.add(key)
                est_vals = _values_for(st, "est", max_len)
                sd_vals = _values_for(st, "sd", max_len)
                lo_vals = _values_for(st, "p2_5", max_len)
                hi_vals = _values_for(st, "p97_5", max_len)
                for idx in range(max_len):
                    pname = f"{key}{idx + 1}"
                    est = est_vals[idx]
                    sd = sd_vals[idx]
                    lo = lo_vals[idx]
                    hi = hi_vals[idx]
                    params[pname] = {
                        "mean": est,
                        "std": sd,
                        "q05": lo,
                        "q50": est,
                        "q95": hi,
                    }
                    names.append(pname)
                    means.append(est)
                    stds.append(sd)
                    q05.append(lo)
                    q95.append(hi)

        for name, st in self.stats.items():
            if name in processed:
                continue
            # Skip flat p-indexed aliases if grouped blocks were detected to avoid duplicates
            if max_len > 0 and isinstance(name, str) and name.startswith("p") and name[1:].isdigit():
                continue
            est = _to_float(st.get("est", np.nan)) if isinstance(st, dict) else float("nan")
            sd = _to_float(st.get("sd", np.nan)) if isinstance(st, dict) else float("nan")
            lo = None
            hi = None
            if isinstance(st, dict):
                lo = st.get("p2_5")
                if lo is None:
                    lo = st.get("p2.5")
                hi = st.get("p97_5")
                if hi is None:
                    hi = st.get("p97.5")
            lo_f = _to_float(lo)
            hi_f = _to_float(hi)
            params[name] = {
                "mean": est,
                "std": sd,
                "q05": lo_f,
                "q50": est,
                "q95": hi_f,
            }
            names.append(name)
            means.append(est)
            stds.append(sd)
            q05.append(lo_f)
            q95.append(hi_f)

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


def _spawn_rng_streams(seed: Optional[int], n: int):
    """
    Deterministic per-draw RNGs independent of worker count/scheduling.
    """
    if seed is None:
        ss = np.random.SeedSequence()
    else:
        ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(int(n))
    return [np.random.Generator(np.random.PCG64(s)) for s in children]


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
    workers: Optional[int] = 0,
    alpha: float = 0.05,
    center_residuals: bool = True,
    return_band: bool = False,
    jitter: Optional[float] = None,
) -> UncertaintyResult:
    """
    Residual bootstrap with robust refitting and clear diagnostics.

    Notes
    -----
    * Determinism: when ``seed`` is provided, a single RNG stream is used.
      Reproducibility is guaranteed only for single-threaded execution.
    * Strict refit: callers can set ``fit_ctx['strict_refit'] = True`` to forbid
      the linearized/fast fallback path, ensuring every successful draw results
      from a full non-linear refit.
    * Diagnostics: the returned diagnostics map contains a ``bootstrap_mode``
      key (``"refit"`` or ``"linearized"``) describing which pathway produced
      the accepted draws.
    """
    t0 = time.time()
    fit_ctx = dict(fit_ctx or {})
    # allow callers to pass normalized jitter explicitly
    if jitter is not None:
        try:
            fit_ctx["bootstrap_jitter"] = float(jitter)
        except Exception:
            pass
    fit = fit_ctx
    strict_refit = bool(fit.get("strict_refit", False))
    progress_cb = fit.get("progress_cb")
    abort_evt = fit.get("abort_event")
    peaks_obj = fit.get("peaks") or fit.get("peaks_out") or fit.get("peaks_in")
    solver = fit.get("solver", "classic")
    fit.setdefault("solver", solver)
    baseline = fit.get("baseline", None)
    mode = fit.get("mode", "add") or "add"
    share_fwhm = bool(fit.get("lmfit_share_fwhm", False))
    share_eta = bool(fit.get("lmfit_share_eta", False))
    jitter_scale = float(fit.get("bootstrap_jitter", 0.0))
    allow_linear = bool(fit.get("allow_linear_fallback", True))

    # Resolve worker count for DRAW phase
    if workers not in (None, False):
        _workers_req = workers
    else:
        _workers_req = fit.get("unc_workers", None)
    try:
        draw_workers = max(0, min(int(_workers_req or 0), os.cpu_count() or 1))
    except Exception:
        draw_workers = 0

    # Resolve worker count for BAND predict_full (optional separate knob)
    _band_req = fit.get("unc_band_workers", fit.get("unc_workers", None))
    try:
        band_workers = max(0, min(int(_band_req or 0), os.cpu_count() or 1))
    except Exception:
        band_workers = 0

    # GPU toggle: fit_ctx flag OR env var PEAKFIT_USE_GPU=1/true/yes
    def _env_true(s: str) -> bool:
        s = (s or "").strip().lower()
        return s in ("1", "true", "yes", "on")

    use_gpu = bool(fit.get("unc_use_gpu", False)) or _env_true(os.getenv("PEAKFIT_USE_GPU", "0"))

    if strict_refit and not (callable(fit.get("refit")) or fit.get("solver") is not None):
        raise RuntimeError(
            "bootstrap_ci(strict_refit=True) requires fit_ctx['solver'] or fit_ctx['refit']."
        )

    # Inputs
    theta = np.asarray(theta, float)
    residual = np.asarray(residual, float)
    n = int(residual.size)
    J = np.asarray(jacobian, float)

    x_src = fit.get("x_all", x_all)
    y_src = fit.get("y_all", y_all)
    if x_src is None or y_src is None:
        raise ValueError("x_all and y_all required for residual bootstrap")
    x_all = np.atleast_1d(np.asarray(x_src, float))[:n]
    y_arr = np.atleast_1d(np.asarray(y_src, float))
    if y_arr.size != n:
        y_arr = y_arr[:n]
    y_all = y_arr if y_arr.size == n else None

    # Center residuals if requested
    r = residual - residual.mean() if center_residuals else residual.copy()

    # Baseline model prediction (prefer predict_full to avoid loss-mode mismatch)
    diag_notes: List[str] = []
    if callable(predict_full):
        try:
            y_hat = np.asarray(predict_full(theta), float)
        except Exception as e:
            diag_notes.append(repr(e))
            base = y_all if y_all is not None else np.zeros_like(residual)
            y_hat = base - residual
    else:
        base = y_all if y_all is not None else np.zeros_like(residual)
        y_hat = base - residual
    if y_hat.size != n:
        y_hat = y_hat[:n]
    if y_all is None:
        y_all = (y_hat + residual)[:n]
    else:
        y_all = np.atleast_1d(np.asarray(y_all, float))
        if y_all.size != n:
            y_all = (y_hat + residual)[:n]

    # If we have no peaks, fall back to asymptotic CI to avoid crashy path
    if not peaks_obj:
        ymodel = predict_full if callable(predict_full) else (lambda _th: np.asarray(y_all, float))
        res_asym = asymptotic_ci(theta, residual, J, ymodel, alpha=alpha)

        band = res_asym.band if (return_band and predict_full is not None) else None

        diag = dict(res_asym.diagnostics)
        diag.update({
            "aborted": False,
            "reason": "no-peaks",
            "n_boot": int(n_boot),
            "n_success": 0,
            "n_fail": 0,
            "band_source": "asymptotic" if band is not None else None,
            "band_reason": None if band is not None else "missing model",
        })
        diag.setdefault("pct_at_bounds", None)
        diag.setdefault("pct_at_bounds_units", "percent")
        diag.setdefault("n_linear_fallback", 0)
        diag.setdefault("linear_lambda", None)
        diag.setdefault("refit_errors", [])
        diag["bootstrap_mode"] = "refit"

        return UncertaintyResult(
            method="bootstrap",
            label="Bootstrap",
            stats=res_asym.stats,
            diagnostics=diag,
            band=band,
        )

    # --- Robust refit adapter (supports both signatures) ---
    from inspect import signature
    from . import fit_api as _fit_api
    try:
        _sig = signature(_fit_api.run_fit_consistent)
    except Exception as e:
        diag_notes.append(repr(e))
        _sig = None

    def _mk_cfg():
        cfg = {"solver": solver, "mode": mode}
        if str(solver).lower().startswith("lmfit"):
            # Only used if lmfit backend honors these flags
            cfg["lmfit_share_fwhm"] = share_fwhm
            cfg["lmfit_share_eta"] = share_eta
        return cfg

    def _robust_refit(theta_init, x, y):
        cfg = _mk_cfg()
        if _sig:
            params = set(_sig.parameters.keys())
            call = {"x": x, "y": y}

            # Supply configuration under the name the function supports
            if "cfg" in params:
                call["cfg"] = cfg
            elif "config" in params:
                call["config"] = cfg

            # support both new spellings for peaks
            if "peaks_in" in params:
                call["peaks_in"] = peaks_obj
            elif "peaks" in params:
                call["peaks"] = peaks_obj

            # pass only kwargs that exist in the function signature
            optional = {
                "baseline": baseline,
                "mode": mode,
                "fit_mask": np.ones_like(x, dtype=bool),
                "rng_seed": None,
                "verbose": False,
                "quick_and_dirty": False,
                "theta_init": theta_init,
                "locked_mask": locked_mask,
                "bounds": bounds,
            }
            for k, v in optional.items():
                if k in params:
                    call[k] = v
            try:
                res = _fit_api.run_fit_consistent(**call)
                th = np.asarray(res.get("theta", theta_init), float)
                ok = bool(res.get("fit_ok", res.get("ok", False)))
                return th, ok
            except Exception:
                # fall through to legacy below
                pass

        # LEGACY fallback: (x, y, cfg_with_peaks_dicts, ...)
        from .data_io import peaks_to_dicts
        cfg_legacy = {**cfg, "peaks": peaks_to_dicts(peaks_obj)}
        res = _fit_api.run_fit_consistent(
            x, y, cfg_legacy,
            theta_init=theta_init, locked_mask=locked_mask,
            bounds=bounds, baseline=baseline
        )
        th = np.asarray(res.get("theta", theta_init), float)
        ok = bool(res.get("fit_ok", res.get("ok", False)))
        return th, ok

    # --- Optional user-supplied refit from fit_ctx (batch path) ---
    user_refit = fit.get("refit", None)
    if callable(user_refit):
        def refit(theta_init, x, y):
            out = user_refit(theta_init, locked_mask, bounds, x, y)
            if isinstance(out, tuple) and len(out) == 2:
                th_new, ok = out
                th_new = np.asarray(th_new, float)
                return th_new, bool(ok)
            # Explicit contract: a bare array return is NOT success
            th_new = np.asarray(out, float)
            return th_new, False
    else:
        refit = _robust_refit

    # Free-parameter mask for optional linear fallback
    theta0 = np.asarray(fit.get("theta0", theta), float)
    P = theta.size
    free_mask = np.ones(P, bool)
    if locked_mask is not None:
        free_mask = ~np.asarray(locked_mask, bool)
    Jf = J[:, free_mask] if (J.ndim == 2) else None
    # Disable linear fallback when parameters are tied (LMFIT) or globally disabled
    if share_fwhm or share_eta or not allow_linear:
        Jf = None
    use_linearized_fast_path = bool(Jf is not None and Jf.size and np.sum(free_mask) > 0)
    if strict_refit or jitter_scale > 0:
        use_linearized_fast_path = False
        Jf = None

    # Bootstrap loop
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")

    rng_streams = _spawn_rng_streams(seed, int(n_boot))
    T_list: List[np.ndarray] = []
    n_success = 0
    n_fail = 0
    linear_fallbacks = 0
    linear_lambda = None
    refit_errors: List[str] = []
    pulse_step = max(1, int(n_boot // 20))
    next_pulse_at = 0
    last_pulse_t = time.monotonic()
    aborted = False

    def _one_draw(b: int):
        rng_local = rng_streams[b]
        # resample residuals
        idx = rng_local.integers(0, n, size=n)
        r_b = r[idx]
        y_b = (y_hat + r_b)

        # jitter start (free params only)
        theta_init = theta0.copy()
        if jitter_scale > 0 and np.any(free_mask):
            step = jitter_scale * np.maximum(np.abs(theta_init), 1.0)
            theta_init[free_mask] += rng_local.normal(0.0, step[free_mask])

        ok = False
        th_new = theta_init
        err_msg = None
        try:
            th_new, ok = refit(theta_init, x_all, y_b)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"

        if not ok and (Jf is None or not Jf.size or np.sum(free_mask) == 0):
            try:
                th_try = theta_init + 1e-6 * (theta0 - theta_init)
                th_new, ok = refit(th_try, x_all, y_b)
            except Exception as e:
                if err_msg is None:
                    err_msg = f"{type(e).__name__}: {e}"

        return b, np.asarray(th_new, float), bool(ok), err_msg

    def _pulse(done_i: int):
        nonlocal next_pulse_at, last_pulse_t
        if progress_cb is not None and (done_i >= next_pulse_at or (time.monotonic() - last_pulse_t) > 0.5):
            try:
                progress_cb(f"Bootstrap: {done_i}/{int(n_boot)}")
            except Exception:
                pass
            last_pulse_t = time.monotonic()
            next_pulse_at = done_i + pulse_step

    if draw_workers > 0:
        with ThreadPoolExecutor(max_workers=draw_workers) as ex:
            futs = {ex.submit(_one_draw, int(b)): int(b) for b in range(int(n_boot))}
            done_cnt = 0
            for f in as_completed(futs):
                if abort_evt is not None:
                    try:
                        if abort_evt.is_set():
                            aborted = True
                            break
                    except Exception:
                        pass
                _, th_new, ok, err_msg = f.result()
                done_cnt += 1
                _pulse(done_cnt)
                if ok and np.all(np.isfinite(th_new)):
                    T_list.append(th_new)
                    n_success += 1
                else:
                    n_fail += 1
                    if err_msg and err_msg not in refit_errors and len(refit_errors) < 5:
                        refit_errors.append(err_msg)
    else:
        for b in range(int(n_boot)):
            if abort_evt is not None:
                try:
                    if abort_evt.is_set():
                        aborted = True
                        break
                except Exception:
                    pass
            _, th_new, ok, err_msg = _one_draw(int(b))
            if ok and np.all(np.isfinite(th_new)):
                T_list.append(th_new)
                n_success += 1
            else:
                n_fail += 1
                if err_msg and err_msg not in refit_errors and len(refit_errors) < 5:
                    refit_errors.append(err_msg)
            _pulse(b + 1)

    if not T_list or len(T_list) < 2:
        raise RuntimeError(f"Insufficient successful bootstrap refits (success={n_success}, fail={n_fail})")

    T = np.vstack(T_list)
    mean = T.mean(axis=0)
    sd = T.std(axis=0, ddof=1)
    qlo = np.quantile(T, alpha/2, axis=0)
    qhi = np.quantile(T, 1 - alpha/2, axis=0)

    names = param_names or [f"p{i}" for i in range(P)]
    stats = {names[i]: {"est": float(mean[i]), "sd": float(sd[i]), "p2.5": float(qlo[i]), "p97.5": float(qhi[i])}
             for i in range(P)}

    # % of successful thetas hitting bounds (useful for diagnosing degeneracy)
    pct_at_bounds = 0.0
    if bounds is not None:
        lo_b, hi_b = bounds
        lo_hit = np.any(np.isclose(T, lo_b, rtol=0, atol=0), axis=1) if (lo_b is not None) else False
        hi_hit = np.any(np.isclose(T, hi_b, rtol=0, atol=0), axis=1) if (hi_b is not None) else False
        pct_at_bounds = float(100.0 * np.mean(lo_hit | hi_hit))
    # pct_at_bounds is expressed in percent (0-100)

    # Optional percentile band (subsample to control memory)
    band = None
    band_reason = None
    diagnostics: Dict[str, object] = {}
    if return_band:
        if predict_full is None or len(T_list) < BOOT_BAND_MIN_SAMPLES:
            band_reason = "missing model or insufficient samples"
        else:
            T = np.vstack(T_list)
            max_use = min(T.shape[0], 4096)
            sel = np.linspace(0, T.shape[0]-1, max_use, dtype=int)

            def _eval_one(i: int):
                yhat = predict_full(T[i])
                try:
                    import cupy as cp  # optional
                    if "cupy" in str(type(yhat)).lower():
                        return yhat  # keep CuPy on GPU
                except Exception:
                    pass
                return np.asarray(yhat)

            Y_list: List[np.ndarray] = []
            if band_workers > 0:
                with ThreadPoolExecutor(max_workers=band_workers) as ex:
                    futs = {ex.submit(_eval_one, int(i)): int(i) for i in sel}
                    for f in as_completed(futs):
                        try:
                            Y_list.append(f.result())
                        except Exception as e:
                            diag_notes.append(repr(e))
            else:
                for i in sel:
                    try:
                        Y_list.append(_eval_one(int(i)))
                    except Exception as e:
                        diag_notes.append(repr(e))

            band_backend = "numpy"
            try:
                import cupy as cp  # optional
                if use_gpu:
                    # If any element already on GPU, keep it; else copy to GPU
                    if any("cupy" in str(type(y)).lower() for y in Y_list):
                        Y = cp.stack([y if isinstance(y, cp.ndarray) else cp.asarray(y) for y in Y_list], axis=0)
                    else:
                        Y = cp.asarray(np.stack(Y_list, axis=0))
                    lo = cp.quantile(Y, float(alpha/2), axis=0)
                    hi = cp.quantile(Y, float(1 - alpha/2), axis=0)
                    lo = cp.asnumpy(lo); hi = cp.asnumpy(hi)
                    band_backend = "cupy"
                else:
                    Y = np.stack(Y_list, axis=0)
                    lo = np.quantile(Y, alpha/2, axis=0)
                    hi = np.quantile(Y, 1 - alpha/2, axis=0)
            except Exception:
                Y = np.stack(Y_list, axis=0)
                lo = np.quantile(Y, alpha/2, axis=0)
                hi = np.quantile(Y, 1 - alpha/2, axis=0)
                band_backend = "numpy"

            band = (x_all, lo, hi)
            diagnostics["band_backend"] = band_backend
            diagnostics["band_workers_used"] = int(band_workers) if band_workers > 0 else None
            try:
                import cupy as cp  # optional
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    diag = {
        "B": int(n_boot),
        "n_boot": int(n_boot),
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "n_linear_fallback": int(linear_fallbacks),
        "linear_lambda": float(linear_lambda) if linear_lambda is not None else None,
        "seed": seed,
        "pct_at_bounds": pct_at_bounds,
        "pct_at_bounds_units": "percent",
        "aborted": bool(aborted or ((abort_evt.is_set()) if abort_evt is not None else False)),
        "runtime_s": float(time.time() - t0),
        "theta_jitter_scale": float(jitter_scale),
        "band_source": "bootstrap-percentile" if band is not None else None,
        "band_reason": band_reason,
        "refit_errors": refit_errors,
        "alpha": float(alpha),
    }
    if diag_notes:
        diag["notes"] = diag_notes
    # Merge any per-band diagnostics (e.g. workers_used, band_backend)
    diag.update(diagnostics)
    diag["bootstrap_mode"] = "linearized" if use_linearized_fast_path else "refit"
    diag["draw_workers_used"] = int(draw_workers) if draw_workers > 0 else None

    return UncertaintyResult(method="bootstrap", label="Bootstrap", stats=stats, diagnostics=diag, band=band)
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
    seed=None, workers: Optional[int] = None, return_band=True, prior_sigma="half_cauchy"
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
    diag_notes: List[str] = []
    # Significance level for credible intervals (used in per-parameter quantiles)
    alpha = float((fit_ctx or {}).get("alpha", 0.05))
    # Optional tying (LMFIT "share FWHM/eta"): collapse tied params to one free scalar
    share_fwhm = bool((fit_ctx or {}).get("lmfit_share_fwhm", False))
    share_eta = bool((fit_ctx or {}).get("lmfit_share_eta", False))
    # Pull bounds/locks from fit_ctx when not explicitly provided
    if bounds is None and fit_ctx and "bounds" in fit_ctx:
        bounds = fit_ctx.get("bounds")
    if locked_mask is None and fit_ctx and "locked_mask" in fit_ctx:
        locked_mask = fit_ctx.get("locked_mask")
    locked_eff = locked_mask.copy() if locked_mask is not None else np.zeros(theta_hat.size, bool)
    tie_groups: list[tuple[int, list[int]]] = []
    if share_fwhm:
        idx = [4 * i + 2 for i in range(theta_hat.size // 4)]
        leader = next((j for j in idx if not locked_eff[j]), idx[0])
        for j in idx:
            if j != leader:
                locked_eff[j] = True
        tie_groups.append((leader, idx))
    if share_eta:
        idx = [4 * i + 3 for i in range(theta_hat.size // 4)]
        leader = next((j for j in idx if not locked_eff[j]), idx[0])
        for j in idx:
            if j != leader:
                locked_eff[j] = True
        tie_groups.append((leader, idx))
    free_idx = np.where(~np.asarray(locked_eff, bool))[0]
    P_free = int(free_idx.size)
    th_free = theta_hat[free_idx]
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

    def _project_full(th_f):
        th_full = theta_hat.copy()
        if P_free:
            th_full[free_idx] = th_f
        # replicate leaders to their tied group members
        for leader, group in tie_groups:
            th_full[group] = th_full[leader]
        return th_full

    def log_likelihood(th_f, log_sigma):
        th_full = _project_full(th_f)
        if bounds is not None:
            lo_b, hi_b = bounds
            if (lo_b is not None and np.any(th_full < lo_b)) or (
                hi_b is not None and np.any(th_full > hi_b)
            ):
                return -np.inf
        mu = pred(th_full)
        if not np.all(np.isfinite(mu)): return -np.inf
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
    # Sane defaults if caller passes 0/None
    if n_burn is None or n_burn <= 0:
        n_burn = 1000
    if n_steps is None or n_steps <= 0:
        n_steps = 4000
    n_burn = int(n_burn)
    n_steps = int(n_steps)
    if n_walkers is None:
        n_walkers = max(4 * dim, 16)
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
    if workers not in (None, False):
        val_workers = workers
    else:
        val_workers = (fit_ctx or {}).get("unc_workers", 0)
    try:
        w = max(0, min(int(val_workers or 0), os.cpu_count() or 1))
    except Exception:
        w = 0
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
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

    # Early abort/minimal mode: skip heavy post-processing
    if abort_evt is not None and getattr(abort_evt, "is_set", lambda: False)():
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics={"aborted": True, "n_draws": 0, "band_source": None},
            band=None,
        )

    chain_post = sampler.get_chain(discard=n_burn, thin=thin, flat=False)  # (steps, walkers, dim)
    acc_frac = float(np.mean(sampler.acceptance_fraction))
    if chain_post.ndim != 3:
        chain_post = np.asarray(chain_post).reshape((-1, n_walkers, dim))
    n_post = chain_post.shape[0]  # number of post-burn/thinned steps per walker
    if n_post <= 0:
        # Avoid "index -1" crashes when burn==total or abort mid-burn
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics={"aborted": True, "n_draws": 0, "accept_frac_mean": acc_frac, "band_source": None},
            band=None,
        )
    n_samp = n_post * n_walkers

    flat = chain_post.reshape(-1, dim)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    n_samp = flat.shape[0]
    if n_samp < 2:
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics={"aborted": True, "n_draws": 0, "accept_frac_mean": acc_frac, "band_source": None},
            band=None,
        )
    th_draws = flat[:, :P_free]
    log_sigma_draws = flat[:, -1]
    sigma_draws = np.exp(log_sigma_draws)

    T_full = np.tile(theta_hat, (th_draws.shape[0], 1))
    if P_free:
        T_full[:, free_idx] = th_draws
    # Apply ties on all draws so the full parameter order is preserved
    for leader, group in tie_groups:
        T_full[:, group] = T_full[:, [leader]]

    def _nanstd_safe(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size < 2:
            return float(np.nanstd(a, ddof=0))
        return float(np.nanstd(a, ddof=1))

    P = int(T_full.shape[1])
    n_pk = P // 4 if P >= 4 else 0

    def _col(col: int) -> np.ndarray:
        return T_full[:, col]

    def _stat_vec(offset: int) -> Dict[str, List[float]]:
        vals: List[float] = []
        sds: List[float] = []
        lo2: List[float] = []
        hi97: List[float] = []
        for i in range(n_pk):
            samp = _col(4 * i + offset)
            finite = np.isfinite(samp)
            if finite.any():
                sf = samp[finite]
                vals.append(float(np.nanmedian(sf)))
                sds.append(float(_nanstd_safe(sf)))
                lo2.append(float(np.nanpercentile(sf, 100.0 * (alpha / 2))))
                hi97.append(float(np.nanpercentile(sf, 100.0 * (1.0 - alpha / 2))))
            else:
                nan = float("nan")
                vals.append(nan)
                sds.append(nan)
                lo2.append(nan)
                hi97.append(nan)
        return {"est": vals, "sd": sds, "p2_5": lo2, "p97_5": hi97}

    param_stats: Dict[str, Dict[str, Any]] = {
        "center": _stat_vec(0),
        "height": _stat_vec(1),
        "fwhm": _stat_vec(2),
        "eta": _stat_vec(3),
    }

    finite = np.isfinite(sigma_draws)
    if finite.any():
        sigma_fin = sigma_draws[finite]
        param_stats["sigma"] = {
            "est": float(np.nanmedian(sigma_fin)),
            "sd": _nanstd_safe(sigma_fin),
            "p2_5": float(np.nanpercentile(sigma_fin, 100.0 * (alpha / 2))),
            "p97_5": float(np.nanpercentile(sigma_fin, 100.0 * (1.0 - alpha / 2))),
        }
    else:
        param_stats["sigma"] = {
            "est": float("nan"),
            "sd": float("nan"),
            "p2_5": float("nan"),
            "p97_5": float("nan"),
        }

    def _scalar_stats_for(samples: np.ndarray) -> Dict[str, float]:
        finite = np.isfinite(samples)
        if finite.any():
            sf = samples[finite]
            return {
                "est": float(np.nanmedian(sf)),
                "sd": float(_nanstd_safe(sf)),
                "p2_5": float(np.nanpercentile(sf, 100.0 * (alpha / 2))),
                "p97_5": float(np.nanpercentile(sf, 100.0 * (1.0 - alpha / 2))),
            }
        nan = float("nan")
        return {"est": nan, "sd": nan, "p2_5": nan, "p97_5": nan}

    default_names = [f"p{i}" for i in range(P)]
    custom_names = list(param_names) if param_names is not None else []
    for idx in range(P):
        stats_flat = _scalar_stats_for(T_full[:, idx])
        param_stats[default_names[idx]] = stats_flat
        if idx < len(custom_names):
            key = str(custom_names[idx])
            if key and key not in param_stats:
                param_stats[key] = dict(stats_flat)

    # Build post-burn/thinned chain view for diagnostics
    try:
        chain = sampler.get_chain(flat=False)  # (total_steps, n_walkers, dim)
        post = chain[-int(n_steps):, :, :]     # keep steps after burn
        if int(thin) > 1:
            post = post[::int(thin), :, :]
    except Exception as e:
        diag_notes.append(repr(e))
        post = None

    ess_min = float("nan")
    rhat_max = float("nan")
    mcse16 = mcse50 = mcse84 = float("nan")

    if post is not None and post.size:
        try:
            ess = ess_autocorr(post)               # expects (steps, chains, dim)
            ess_min = float(np.nanmin(np.asarray(ess)))
        except Exception as e:
            diag_notes.append(repr(e))
        try:
            rhat = rhat_split(post)                # expects (steps, chains, dim)
            rhat_max = float(np.nanmax(np.asarray(rhat)))
        except Exception as e:
            diag_notes.append(repr(e))
        # Conservative MCSE for marginal quantiles via per-walker quantile std
        try:
            W = post.shape[1]

            def _mcse_q(q):
                q_walker = np.nanquantile(post, q, axis=0)   # (walkers, dim)
                return float(
                    np.nanmax(
                        np.nanstd(q_walker, axis=0, ddof=1)
                        / max(np.sqrt(W), 1.0)
                    )
                )

            mcse16 = _mcse_q(0.16)
            mcse50 = _mcse_q(0.50)
            mcse84 = _mcse_q(0.84)
        except Exception as e:
            diag_notes.append(repr(e))

    diag = {
        "n_draws": int(n_samp),
        "ess_min": float(ess_min),
        "rhat_max": float(rhat_max),
        "accept_frac_mean": acc_frac,
        "seed": seed,
        "aborted": bool(aborted),
        "band_source": None,
        "mcse": {"q16": float(mcse16), "q50": float(mcse50), "q84": float(mcse84)},
    }
    if diag_notes:
        diag["notes"] = diag_notes

    # Never compute bands for Bayesian (disabled)
    return UncertaintyResult(
        method="bayesian",
        label="Bayesian (MCMC)",
        stats=param_stats,
        diagnostics=diag,
        band=None,
    )
