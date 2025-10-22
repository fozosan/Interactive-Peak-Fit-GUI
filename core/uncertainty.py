"""Light-weight uncertainty estimators used in the tests.

The real project exposes a rather feature rich uncertainty module.  For the
kata we implement a very small subset that mimics the public surface of the
original functions.  The goal is API compatibility and deterministic behaviour
rather than ultimate statistical rigour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:  # numpy.typing is available in NumPy >= 1.20
    from numpy.typing import ArrayLike
except Exception:  # pragma: no cover
    ArrayLike = Any  # type: ignore

import logging
import warnings
import time
import math
import os
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# -----------------------
# Internal helper utilities
# -----------------------


def _norm_solver_and_sharing(fit: Dict, kw: Dict) -> Tuple[str, bool, bool]:
    """Normalize solver name and sharing flags in a single place."""
    solver = str(
        (fit or {}).get("solver")
        or (fit or {}).get("solver_used")
        or kw.get("solver")
        or "modern_trf"
    ).lower()
    # Generic share flags (solver-agnostic)
    share_fwhm = bool((fit or {}).get("share_fwhm", False))
    share_eta = bool((fit or {}).get("share_eta", False))
    # LMFit-specific share flags only apply if LMFit is actually used
    if solver.startswith("lmfit"):
        share_fwhm = share_fwhm or bool((fit or {}).get("lmfit_share_fwhm", False))
        share_eta = share_eta or bool((fit or {}).get("lmfit_share_eta", False))
    return solver, share_fwhm, share_eta


def _build_residual_vector(
    residual: np.ndarray,
    y_all: Optional[np.ndarray],
    y_hat: Optional[np.ndarray],
    mode: str,
    center: bool,
) -> np.ndarray:
    """Prefer RAW residuals (y - y_hat) to avoid weighting artifacts; fallback safely."""
    mode = str(mode or "raw").lower()
    if mode == "raw" and (y_all is not None) and (y_hat is not None):
        r = np.asarray(y_all, float).reshape(-1) - np.asarray(y_hat, float).reshape(-1)
        if center:
            r = r - float(np.mean(r))
        return r
    # fallback to provided residuals (may be weighted upstream)
    r = np.asarray(residual, float).reshape(-1).copy()
    if center:
        r = r - float(np.mean(r))
    return r


def _relabel_by_center(theta_new: np.ndarray, theta_ref: np.ndarray, block: int = 4) -> np.ndarray:
    """Reorder parameter blocks so peak identity is consistent w.r.t. reference centers."""
    try:
        ref_c = np.asarray(theta_ref[0::block], float)
        cur_c = np.asarray(theta_new[0::block], float)
        used = set()
        order: List[int] = []
        for target in ref_c:
            j_best, d_best = None, float("inf")
            for j, cval in enumerate(cur_c):
                if j in used:
                    continue
                d = abs(float(cval) - float(target))
                if d < d_best:
                    d_best, j_best = d, j
            if j_best is None:
                j_best = next(i for i in range(len(cur_c)) if i not in used)
            used.add(j_best)
            order.append(int(j_best))
        th_re = np.empty_like(theta_new)
        for new_i, old_i in enumerate(order):
            th_re[block * new_i : block * (new_i + 1)] = theta_new[block * old_i : block * (old_i + 1)]
        return th_re
    except Exception:
        return theta_new


def _validate_vector_length(name: str, arr: np.ndarray, n: int) -> None:
    if arr is None:
        return
    if np.asarray(arr).size != int(n):
        # Be explicit about the contract: predictors must live on the fit window (len(x_all) == n).
        raise ValueError(
            f"{name} produced vector of wrong size (got {np.asarray(arr).size}, want {n}); "
            f"provide a predictor defined on the fit window (len(x_all) == {n})."
        )

try:
    from scipy.optimize import nnls as _nnls
except Exception:  # pragma: no cover - optional dependency
    _nnls = None

from .fit_api import _vp_design_columns
from .mcmc_utils import ess_autocorr, rhat_split
from infra import performance

__all__ = [
    "asymptotic_ci",
    "bootstrap_ci",
    "bayesian_ci",
    "UncertaintyResult",
    "finite_diff_jacobian",
    "BAND_SKIP_DIAG_DISABLED",
    "BAND_SKIP_DIAG_UNHEALTHY",
    "BAND_SKIP_INSUFF_DRAWS",
    "BAND_SKIP_HARD_FAILURE",
    "BAND_SKIP_ABORTED",
]

log = logging.getLogger(__name__)

BOOT_BAND_MIN_SAMPLES = 16
# Minimum number of total MCMC draws required before attempting a tiny band
# (kept small to stay "tiny" and fast). Adjust if needed.
BAYES_BAND_MIN_DRAWS = 32

# Common skip reasons (keep strings stable across GUI/Batch/Tests)
BAND_SKIP_DIAG_DISABLED = "diagnostics_disabled"
BAND_SKIP_DIAG_UNHEALTHY = "diagnostics_unhealthy"
BAND_SKIP_INSUFF_DRAWS = "insufficient_draws"
BAND_SKIP_HARD_FAILURE = "hard_failure"
BAND_SKIP_ABORTED = "aborted"


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
            if val is None:
                if key == "p2_5":
                    val = st.get("p2.5")
                elif key == "p97_5":
                    val = st.get("p97.5")
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

        stats_obj = self.stats
        # Build a working map and synthesize vector blocks from per-peak rows if present.
        if isinstance(stats_obj, dict):
            stats_map: Dict[str, Any] = dict(stats_obj)
        else:
            stats_map = {}

        rows_from_stats: list[Dict[str, Any]] = []
        maybe_rows = stats_map.get("rows") if isinstance(stats_map, dict) else None
        if isinstance(maybe_rows, (list, tuple)):
            rows_from_stats = [dict(r) for r in maybe_rows if isinstance(r, dict)]
        elif isinstance(stats_obj, (list, tuple)):
            rows_from_stats = [dict(r) for r in stats_obj if isinstance(r, dict)]

        if rows_from_stats:
            # Aggregate rows -> vector blocks and ALWAYS mirror dotted/underscored quantiles per row.
            agg: Dict[str, Dict[str, list[Any]]] = {}
            for row in rows_from_stats:
                for key in block_keys:
                    blk = row.get(key)
                    dest = agg.setdefault(key, {})
                    if isinstance(blk, dict):
                        # Scalarize and append est/sd (or NaN if missing)
                        est_v = _to_float(blk.get("est"))
                        sd_v = _to_float(blk.get("sd"))
                        dest.setdefault("est", []).append(est_v)
                        dest.setdefault("sd", []).append(sd_v)
                        # Mirror quantiles so both alias styles have identical per-peak lengths
                        qlo = blk.get("p2_5", blk.get("p2.5"))
                        qhi = blk.get("p97_5", blk.get("p97.5"))
                        qlo_f = _to_float(qlo)
                        qhi_f = _to_float(qhi)
                        dest.setdefault("p2_5", []).append(qlo_f)
                        dest.setdefault("p2.5", []).append(qlo_f)
                        dest.setdefault("p97_5", []).append(qhi_f)
                        dest.setdefault("p97.5", []).append(qhi_f)
                    else:
                        # Non-dict row → pad all fields to keep lengths aligned
                        nan = float("nan")
                        dest.setdefault("est", []).append(nan)
                        dest.setdefault("sd", []).append(nan)
                        dest.setdefault("p2_5", []).append(nan)
                        dest.setdefault("p2.5", []).append(nan)
                        dest.setdefault("p97_5", []).append(nan)
                        dest.setdefault("p97.5", []).append(nan)
            for key, block in agg.items():
                # Ensure dotted/underscored quantile aliases both exist (already mirrored above).
                stats_map[key] = block

        max_len = 0
        for key in block_keys:
            st = stats_map.get(key) if isinstance(stats_map, dict) else None
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
                st = stats_map.get(key) if isinstance(stats_map, dict) else None
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

        if isinstance(stats_map, dict):
            items_iter = stats_map.items()
        else:
            items_iter = []
        for name, st in items_iter:
            if name in processed:
                continue
            if name == "rows":
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
    locked_mask: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **_ignored: Any,
) -> UncertaintyResult:
    """Return asymptotic parameter statistics and a prediction band.

    ``residual`` and ``jacobian`` may be arrays already evaluated at
    ``theta_hat`` or callables accepting a parameter vector.  Extra keyword
    arguments are ignored for backwards compatibility.
    """

    theta = np.asarray(theta_hat, float)
    P = int(theta.size)

    def _canon_locked_mask(mask, P):
        if mask is None:
            return None
        lm = np.asarray(mask, bool).ravel()
        if lm.size == P:
            return lm
        if P % 4 == 0 and lm.size == (P // 4):
            return np.repeat(lm, 4)
        return None

    locked_arr = _canon_locked_mask(locked_mask, P)
    locked_mask = locked_arr if locked_arr is not None else None
    free_mask = np.ones(P, bool) if locked_mask is None else ~locked_mask

    lo_raw = hi_raw = None
    if bounds is not None:
        try:
            lo_raw, hi_raw = bounds
        except Exception:
            lo_raw = hi_raw = None

    def _norm_bounds_component(raw: Optional[ArrayLike], fill_value: float) -> np.ndarray:
        if raw is None:
            return np.full(P, fill_value, dtype=float)
        try:
            arr = np.asarray(raw, float).reshape(-1)
        except Exception:
            return np.full(P, fill_value, dtype=float)
        if arr.size != P:
            return np.full(P, fill_value, dtype=float)
        return arr

    lo = _norm_bounds_component(lo_raw, -np.inf)
    hi = _norm_bounds_component(hi_raw, np.inf)

    r = residual(theta) if callable(residual) else np.asarray(residual, float)
    J = jacobian(theta) if callable(jacobian) else np.asarray(jacobian, float)

    m = int(J.shape[0])
    n_free = int(np.sum(free_mask))
    dof = max(m - n_free, 1)

    Jf = J[:, free_mask] if n_free > 0 else J[:, :0]
    rss = float(np.dot(r, r))
    s2 = rss / dof

    if n_free > 0:
        JTJ = Jf.T @ Jf
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
        cov_free = s2 * JTJ_inv
    else:
        cond = 0.0
        cov_free = np.zeros((0, 0))

    cov = np.zeros((P, P))
    if n_free > 0:
        cov[np.ix_(free_mask, free_mask)] = cov_free

    std = np.zeros(P)
    if n_free > 0:
        std[free_mask] = np.sqrt(np.maximum(np.diag(cov_free), 0.0))
    z = _z_value(alpha)
    ci_lo = theta - z * std
    ci_hi = theta + z * std
    ci_lo = np.maximum(ci_lo, lo)
    ci_hi = np.minimum(ci_hi, hi)

    y0 = np.asarray(ymodel_fn(theta), float)
    G = finite_diff_jacobian(ymodel_fn, theta)
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    band_std = np.sqrt(np.maximum(var, 0.0))
    band_lo = y0 - z * band_std
    band_hi = y0 + z * band_std
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

    def _add_percentile_aliases_inplace(stats_map: Dict[str, Dict[str, Any]]) -> None:
        for _k, blk in list(stats_map.items()):
            if not isinstance(blk, dict):
                continue
            if "p2.5" in blk and "p2_5" not in blk:
                blk["p2_5"] = blk["p2.5"]
            if "p97.5" in blk and "p97_5" not in blk:
                blk["p97_5"] = blk["p97.5"]
            if "p2_5" in blk and "p2.5" not in blk:
                blk["p2.5"] = blk["p2_5"]
            if "p97_5" in blk and "p97.5" not in blk:
                blk["p97.5"] = blk["p97_5"]

    _add_percentile_aliases_inplace(stats)

    band = (x, band_lo, band_hi)
    diag: Dict[str, object] = {"alpha": alpha, "param_order": names}
    n_free = int(np.sum(free_mask))
    # Preferred key for downstream consumers; keep legacy alias for compatibility.
    diag.setdefault("n_free_params", n_free)
    diag.setdefault("locked_free_params", n_free)
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
    # ------------------------------------------------------------------
    # Normalize inputs used widely and ensure bounds arrays are defined
    # immediately so every branch can safely reference them.
    # ------------------------------------------------------------------
    theta = np.asarray(theta, float)
    P = int(theta.size)

    fit_ctx = dict(fit_ctx or {})

    bounds_arg = bounds
    bounds_from_ctx = fit_ctx.get("bounds", None)
    bounds_source = "default"

    bounds_local = None
    if bounds_arg is not None:
        bounds_local = bounds_arg
        bounds_source = "arg"
    elif bounds_from_ctx is not None:
        bounds_local = bounds_from_ctx
        bounds_source = "fit_ctx"

    bounds = bounds_local
    # ------------------------------------------------------------------

    t0 = time.time()
    _solver_name, share_fwhm, share_eta = _norm_solver_and_sharing(fit_ctx, {})
    fit_ctx["solver"] = _solver_name
    fit_ctx["share_fwhm"] = share_fwhm
    fit_ctx["share_eta"] = share_eta
    # Accept explicit jitter kw (normalized fraction) for GUI/batch calls
    if jitter is not None:
        try:
            fit_ctx["bootstrap_jitter"] = float(jitter)
        except Exception:
            pass
    fit = fit_ctx
    # Pull constraints from fit_ctx (batch/gui parity)
    bounds_arg = bounds
    bounds_from_ctx = fit.get("bounds", None)
    if bounds_arg is None and bounds_from_ctx is not None:
        bounds = bounds_from_ctx
    else:
        bounds = bounds_arg
    if bounds is not None:
        fit["bounds"] = bounds
    locked_mask = fit.get("locked_mask", locked_mask)
    strict_refit = bool(fit.get("strict_refit", False))
    progress_cb = fit.get("progress_cb")
    abort_evt = fit.get("abort_event")
    peaks_obj = fit.get("peaks") or fit.get("peaks_out") or fit.get("peaks_in")
    fit.setdefault("bootstrap_residual_mode", "raw")
    fit.setdefault("relabel_by_center", True)
    fit.setdefault("center_residuals", bool(center_residuals))
    baseline = fit.get("baseline", None)
    mode = fit.get("mode", "add") or "add"
    jitter_scale = float(fit.get("bootstrap_jitter", 0.0))
    allow_linear = bool(fit.get("allow_linear_fallback", True))

    # Propagate performance threading knobs for downstream bootstrap helpers.
    strategy = str(fit.get("perf_parallel_strategy", "outer"))
    fit["perf_parallel_strategy"] = strategy
    try:
        fit["perf_blas_threads"] = int(fit.get("perf_blas_threads", 0) or 0)
    except Exception:
        fit["perf_blas_threads"] = 0

    diag_perf = {
        "parallel_strategy": strategy,
        "blas_threads": int(fit.get("perf_blas_threads", 0) or 0),
    }
    # Emit breadcrumbs for logs/debugging (no behavior change)
    diag_perf["solver_used"] = str(fit.get("solver", "")).lower()
    diag_perf["share_fwhm"] = bool(fit.get("share_fwhm", False))
    diag_perf["share_eta"] = bool(fit.get("share_eta", False))
    diag_perf["bootstrap_residual_mode"] = str(fit.get("bootstrap_residual_mode", "raw"))

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
        band_workers_requested = max(0, int(_band_req or 0))
    except Exception:
        band_workers_requested = 0
    _cpu = os.cpu_count() or 1
    band_workers_effective = (
        max(1, min(band_workers_requested, _cpu)) if band_workers_requested > 0 else None
    )
    band_workers = (
        band_workers_effective
        if (band_workers_effective is not None and band_workers_effective > 1)
        else None
    )

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
    residual = np.asarray(residual, float).reshape(-1)
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

    # Canonical starting point for jitter (allow caller override via theta0)
    theta0 = np.asarray(fit.get("theta0", theta), float)

    # Baseline model prediction (prefer predict_full to avoid loss-mode mismatch)
    diag_notes: List[str] = []
    y_hat = None
    if callable(predict_full):
        try:
            y_hat = np.asarray(predict_full(theta), float)
        except Exception as e:
            diag_notes.append(repr(e))
            y_hat = None
    if y_hat is not None:
        y_hat = np.asarray(y_hat, float).reshape(-1)
    if y_all is not None:
        y_all = np.asarray(y_all, float).reshape(-1)

    r = _build_residual_vector(
        residual=residual,
        y_all=None if y_all is None else np.asarray(y_all, float),
        y_hat=None if y_hat is None else np.asarray(y_hat, float),
        mode=str(fit.get("bootstrap_residual_mode", "raw")),
        center=bool(fit.get("center_residuals", True)),
    )

    if y_hat is None:
        base = y_all if y_all is not None else np.zeros_like(residual)
        y_hat = np.asarray(base, float).reshape(-1) - r
    _validate_vector_length("predict_full", y_hat, n)
    if y_all is None:
        y_all = (y_hat + r)
    else:
        y_all = np.atleast_1d(np.asarray(y_all, float).reshape(-1))
        _validate_vector_length("y_all", y_all, n)

    # If we have no peaks, fall back to asymptotic CI to avoid crashy path
    if not peaks_obj:
        def _peaks_from_theta(th_like: np.ndarray):
            try:
                from . import peaks as _peaks_mod
            except Exception:
                return []

            arr = np.asarray(th_like, float).ravel()
            if arr.size % 4 != 0 or arr.size == 0:
                return []

            out: list[_peaks_mod.Peak] = []
            for i in range(arr.size // 4):
                j = 4 * i
                try:
                    out.append(
                        _peaks_mod.Peak(
                            center=float(arr[j + 0]),
                            height=float(arr[j + 1]),
                            fwhm=float(arr[j + 2]),
                            eta=float(arr[j + 3]),
                        )
                    )
                except Exception:
                    return []
            return out

        peaks_obj = _peaks_from_theta(theta0)

    user_refit = fit.get("refit", None)
    if not callable(user_refit):
        module_refit = globals().get("refit", None)
        if callable(module_refit):
            user_refit = module_refit

    if not peaks_obj and not callable(user_refit):
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
    from inspect import signature, Parameter
    from . import fit_api as _fit_api
    try:
        _sig = signature(_fit_api.run_fit_consistent)
        allows_var_kw = any(p.kind == Parameter.VAR_KEYWORD for p in _sig.parameters.values())
    except Exception as e:
        diag_notes.append(repr(e))
        _sig = None
        allows_var_kw = True

    def _mk_cfg():
        cfg = {
            "solver": _solver_name,
            "mode": mode,
            # Enforce strict refits unless explicitly overridden
            "no_solver_fallbacks": True,
        }
        if isinstance(fit, dict) and "no_solver_fallbacks" in fit:
            cfg["no_solver_fallbacks"] = bool(fit.get("no_solver_fallbacks"))
        if str(_solver_name).lower().startswith("lmfit"):
            # Only used if lmfit backend honors these flags
            cfg["lmfit_share_fwhm"] = share_fwhm
            cfg["lmfit_share_eta"] = share_eta
        return cfg

    def _robust_refit(theta_init, x, y):
        cfg = _mk_cfg()
        theta_arr = np.asarray(theta_init, float).copy()

        param_names = set(_sig.parameters.keys()) if _sig is not None else None

        def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if allows_var_kw:
                return dict(kwargs)
            if param_names is None:
                return dict(kwargs)
            return {k: v for k, v in kwargs.items() if k in param_names}

        lb_arr = ub_arr = None
        if bounds is not None:
            try:
                lb_arr, ub_arr = bounds
            except Exception:
                lb_arr = ub_arr = None
        if lb_arr is not None:
            lb_arr = np.asarray(lb_arr, float)
        if ub_arr is not None:
            ub_arr = np.asarray(ub_arr, float)
        if (lb_arr is None or ub_arr is None) and peaks_obj:
            def _try_import_pack():
                try:
                    from fit.bounds import pack_theta_bounds as _ptb

                    return _ptb
                except Exception:
                    try:
                        from .fit.bounds import pack_theta_bounds as _ptb  # type: ignore

                        return _ptb
                    except Exception:
                        return None

            _ptb = _try_import_pack()
            if _ptb is not None:
                x_vals = np.asarray(x, float) if x is not None else np.asarray([], float)
                try:
                    solver_opts = dict(fit.get("solver_options", {}) or {})
                except Exception:
                    solver_opts = {}
                derived: Optional[Tuple[np.ndarray, np.ndarray]] = None
                try:
                    _lb, _ub, *_rest = _ptb(peaks_obj)
                    derived = (np.asarray(_lb, float), np.asarray(_ub, float))
                except Exception:
                    pass
                if derived is None:
                    try:
                        _head, (_lb, _ub) = _ptb(peaks_obj, x_vals, solver_opts)
                        derived = (np.asarray(_lb, float), np.asarray(_ub, float))
                    except Exception:
                        pass
                if derived is None:
                    try:
                        _lb, _ub, *_rest = _ptb(peaks_obj, x_vals, solver_opts)
                        derived = (np.asarray(_lb, float), np.asarray(_ub, float))
                    except Exception:
                        pass
                if derived is not None:
                    if lb_arr is None:
                        lb_arr = derived[0]
                    if ub_arr is None:
                        ub_arr = derived[1]
        if lb_arr is not None and ub_arr is not None:
            eps = 1e-12
            gap = np.asarray(ub_arr - lb_arr, float)
            if gap.shape != theta_arr.shape:
                try:
                    gap = np.broadcast_to(gap, theta_arr.shape)
                except Exception:
                    gap = np.zeros_like(theta_arr)
            delta = np.minimum(eps, 0.5 * np.where(np.isfinite(gap), gap, eps))
            lo_eff = np.asarray(lb_arr, float)
            hi_eff = np.asarray(ub_arr, float)
            lo_eff = np.where(np.isfinite(lo_eff), lo_eff + delta, lo_eff)
            hi_eff = np.where(np.isfinite(hi_eff), hi_eff - delta, hi_eff)
            mask_flip = hi_eff < lo_eff
            if np.any(mask_flip):
                mid = 0.5 * (np.asarray(lb_arr, float) + np.asarray(ub_arr, float))
                lo_eff = np.where(mask_flip, mid, lo_eff)
                hi_eff = np.where(mask_flip, mid, hi_eff)
            theta_arr = np.minimum(np.maximum(theta_arr, lo_eff), hi_eff)

        # Prepare peaks with jittered start values so p0 follows theta_init
        peaks_start = peaks_obj
        try:
            if peaks_obj and 4 * len(peaks_obj) == int(theta_arr.size):
                t = theta_arr.ravel()
                ps = copy.deepcopy(peaks_obj)
                for i, pk in enumerate(ps):
                    j = 4 * i
                    pk.center = float(t[j + 0])
                    pk.height = float(t[j + 1])
                    pk.fwhm = float(t[j + 2])
                    pk.eta = float(t[j + 3])
                peaks_start = ps
        except Exception:
            peaks_start = peaks_obj  # best effort, don’t fail the draw

        attempt_errors: List[str] = []

        fit_mask_full = np.ones_like(x, dtype=bool)
        if _sig:
            params = set(_sig.parameters.keys())
            call_base = {"x": x, "y": y}

            # Supply configuration under the name the function supports
            if "cfg" in params:
                call_base["cfg"] = cfg
            elif "config" in params:
                call_base["config"] = cfg
            if "solver" in params:
                call_base["solver"] = _solver_name

            # support both new spellings for peaks
            if "peaks_in" in params:
                call_base["peaks_in"] = peaks_start
            elif "peaks" in params:
                call_base["peaks"] = peaks_start

            # pass only kwargs that exist in the function signature
            optional = {
                "baseline": baseline,
                "mode": mode,
                "fit_mask": fit_mask_full,
                "rng_seed": None,
                "verbose": False,
                "quick_and_dirty": False,
                "locked_mask": locked_mask,
                "bounds": bounds,
            }
            # NOTE(surgical): Respect user-provided fit range. Prefer fit_mask, fall back to legacy mask.
            _m = fit.get("fit_mask", fit.get("mask", None))
            if _m is not None:
                _m_arr = np.asarray(_m, dtype=bool)
                if _m_arr.shape == x.shape:
                    optional["fit_mask"] = _m_arr
            for k, v in optional.items():
                if k in params:
                    call_base[k] = v
            res_main = None
            try:
                res_main = _fit_api.run_fit_consistent(**_filter_kwargs(call_base))
            except Exception as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                res_main = None
            if res_main is not None:
                th_main = np.asarray(res_main.get("theta", theta_arr), float)
                if np.all(np.isfinite(th_main)):
                    return th_main, True

            call_with_theta = dict(call_base)
            call_with_theta["theta_init"] = theta_arr
            if "solver" in params:
                call_with_theta.setdefault("solver", _solver_name)
            try:
                res_theta = _fit_api.run_fit_consistent(**_filter_kwargs(call_with_theta))
            except Exception as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                res_theta = None
            if res_theta is not None:
                th_theta = np.asarray(res_theta.get("theta", theta_arr), float)
                if np.all(np.isfinite(th_theta)):
                    return th_theta, True

        # LEGACY fallback: (x, y, cfg_with_peaks_dicts, ...)
        from .data_io import peaks_to_dicts
        cfg_legacy = {**cfg, "peaks": peaks_to_dicts(peaks_start)}
        common_kwargs = {
            "x": x,
            "y": y,
            "cfg": cfg_legacy,
            "baseline": baseline,
            "mode": mode,
            "fit_mask": fit_mask_full,
            "locked_mask": locked_mask,
            "bounds": bounds,
            "solver": _solver_name,
        }
        variants = [
            {"peaks_in": peaks_start, **common_kwargs},
            common_kwargs,
        ]
        for kwargs in variants:
            res_main = None
            try:
                res_main = _fit_api.run_fit_consistent(**_filter_kwargs(kwargs))
            except Exception as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                res_main = None
            if res_main is not None:
                th_main = np.asarray(res_main.get("theta", theta_arr), float)
                if np.all(np.isfinite(th_main)):
                    return th_main, True

        if param_names is None or "theta_init" in param_names or allows_var_kw:
            kwargs_final = {
                "x": x,
                "y": y,
                "cfg": cfg_legacy,
                "theta_init": theta_arr,
                "locked_mask": locked_mask,
                "bounds": bounds,
                "baseline": baseline,
                "solver": _solver_name,
            }
            if param_names is None or "peaks_in" in (param_names or set()) or allows_var_kw:
                kwargs_final["peaks_in"] = peaks_start
            elif param_names and "peaks" in param_names:
                kwargs_final["peaks"] = peaks_start
            try:
                res = _fit_api.run_fit_consistent(**_filter_kwargs(kwargs_final))
            except Exception as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
            else:
                th = np.asarray(res.get("theta", theta_arr), float)
                ok = np.all(np.isfinite(th))
                if ok:
                    return th, True

        if attempt_errors:
            raise RuntimeError(attempt_errors[-1])
        return np.asarray(theta_arr, float), False

    # --- helper: canonicalize locked mask to parameter-length ---
    def _canon_locked_mask(mask, P):
        if mask is None:
            return None
        lm = np.asarray(mask, bool).ravel()
        if lm.size == P:
            return lm
        if P % 4 == 0 and lm.size == (P // 4):
            return np.repeat(lm, 4)
        return None

    locked_arr = _canon_locked_mask(locked_mask, P)
    locked_mask = locked_arr if locked_arr is not None else None
    free_mask = np.ones(P, bool) if locked_arr is None else ~locked_arr

    # --- Normalize bounds (length-P, float) BEFORE any use ---
    bounds_provided = bounds is not None
    bounds_valid_shape = True
    bounds_unpack_ok = True
    lo_raw = hi_raw = None
    if bounds is not None:
        try:
            lo_raw, hi_raw = bounds
        except Exception:
            bounds_unpack_ok = False
            lo_raw = hi_raw = None

    def _norm_bounds_component(raw, fill):
        if raw is None:
            return np.full(P, fill, dtype=float)
        try:
            arr = np.asarray(raw, float).reshape(-1)
        except Exception:
            return np.full(P, fill, dtype=float)
        if arr.size == P:
            return arr
        # truncate/pad to P
        out = np.full(P, fill, dtype=float)
        n = min(P, int(arr.size))
        if n:
            out[:n] = arr[:n]
        return out

    lo = _norm_bounds_component(lo_raw, -np.inf)
    hi = _norm_bounds_component(hi_raw, np.inf)

    def _raw_size_matches(raw) -> bool:
        if raw is None:
            return True
        try:
            return np.asarray(raw, float).reshape(-1).size == P
        except Exception:
            return False

    if bounds_provided:
        bounds_valid_shape = _raw_size_matches(lo_raw) and _raw_size_matches(hi_raw)
    else:
        bounds_valid_shape = True

    if not bounds_unpack_ok:
        bounds_valid_shape = False

    if bounds_provided and bounds_unpack_ok:
        bounds = (lo, hi)
    elif not bounds_provided:
        bounds = None
    else:
        bounds = None

    # --- Helper uses lo/hi and locked_mask; safe to define now ---
    def _apply_locks_bounds(th: np.ndarray) -> np.ndarray:
        th = np.asarray(th, float).copy()
        th = np.clip(th, lo, hi)
        if locked_mask is not None:
            th[locked_mask] = theta[locked_mask]
        return th

    if bounds is not None:
        try:
            fit["bounds"] = bounds
        except Exception:
            pass

    # --- Optional user-supplied refit from fit_ctx (batch path) ---
    if callable(user_refit):
        from inspect import signature as _inspect_signature

        try:
            _user_sig = _inspect_signature(user_refit)
        except Exception:
            _user_sig = None

        def refit(theta_init, x, y):
            if _user_sig is not None:
                for args in (
                    (theta_init, locked_mask, bounds, x, y),
                    (theta_init, x, y),
                ):
                    try:
                        _user_sig.bind_partial(*args)
                    except TypeError:
                        continue
                    out_local = user_refit(*args)
                    break
                else:
                    out_local = user_refit(theta_init, locked_mask, bounds, x, y)
            else:
                try:
                    out_local = user_refit(theta_init, locked_mask, bounds, x, y)
                except TypeError:
                    out_local = user_refit(theta_init, x, y)

            out = out_local
            if isinstance(out, dict):
                th_dict = out.get("theta", theta_init)
                th_new = _apply_locks_bounds(th_dict)
                ok_flag = bool(out.get("fit_ok", True)) and np.all(np.isfinite(th_new))
                return th_new, ok_flag or (not strict_refit and np.all(np.isfinite(th_new)))
            if isinstance(out, tuple) and len(out) == 2:
                th_new, ok = out
                th_new = _apply_locks_bounds(th_new)
                return th_new, bool(ok) and np.all(np.isfinite(th_new))
            # Bare array return: allow only when strict_refit is disabled and values are finite
            th_new = _apply_locks_bounds(out)
            return th_new, (not strict_refit) and np.all(np.isfinite(th_new))
    else:
        refit = _robust_refit

    Jf = J[:, free_mask] if (J.ndim == 2) else None
    # Disable linear fallback when parameters are tied (LMFIT) or globally disabled
    if share_fwhm or share_eta or not allow_linear:
        Jf = None
    use_linearized_fast_path = bool(Jf is not None and Jf.size and np.sum(free_mask) > 0)
    if strict_refit or jitter_scale > 0:
        use_linearized_fast_path = False
        Jf = None

    refit_impl = refit

    def refit(theta_init, x, y):
        theta_init = _apply_locks_bounds(theta_init)
        th_new, ok = refit_impl(theta_init, x, y)
        th_new = _apply_locks_bounds(th_new)
        return th_new, bool(ok)

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
    MAX_CAPTURED_REFIT_ERRORS = 64

    def _record_refit_error(msg: Optional[str]) -> None:
        if not msg:
            return
        if len(refit_errors) < MAX_CAPTURED_REFIT_ERRORS:
            refit_errors.append(str(msg))

    pulse_step = max(1, int(n_boot // 20))
    next_pulse_at = 0
    last_pulse_t = time.monotonic()
    aborted = False

    diag_failure = {
        "bootstrap_mode": "linearized" if use_linearized_fast_path else "refit",
        "refit_errors": [],
        "linear_fallbacks": 0,
        "linear_lambda_last": None,
    }

    def _one_draw(b: int):
        nonlocal linear_fallbacks, linear_lambda
        nonlocal diag_failure
        rng_local = rng_streams[b]
        # resample residuals
        idx = rng_local.integers(0, n, size=n)
        r_b = r[idx]
        y_b = (y_hat + r_b)

        # jitter start (free params only)
        theta_init = theta0.copy()
        jitter_applied = False
        jitter_rms_local = 0.0
        jitter_reason = None
        if jitter_scale > 0:
            if np.any(free_mask):
                step = jitter_scale * np.maximum(np.abs(theta_init), 1.0)
                eps = rng_local.normal(0.0, step)
                eps = np.asarray(eps, float)
                try:
                    if eps.shape != theta_init.shape:
                        eps = np.broadcast_to(eps, theta_init.shape)
                except Exception:
                    eps = np.zeros_like(theta_init)
                eps = eps * free_mask.astype(float)
                theta_init = theta_init + eps
                jitter_vals = eps[free_mask]
                jitter_applied = bool(jitter_vals.size and np.any(np.abs(jitter_vals) > 0))
                if jitter_vals.size:
                    jitter_rms_local = float(np.sqrt(np.mean(np.square(jitter_vals))))
            else:
                jitter_reason = "no-free-params"

        theta_init = _apply_locks_bounds(theta_init)

        ok = False
        th_new = theta_init
        err_msg = None
        try:
            th_new, ok = refit(theta_init, x_all, y_b)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
        else:
            if ok and bool(fit.get("relabel_by_center", True)):
                th_new = _relabel_by_center(np.asarray(th_new, float), theta0, block=4)
                th_new = _apply_locks_bounds(th_new)

        if not ok and (Jf is None or not Jf.size or np.sum(free_mask) == 0):
            try:
                th_try = theta_init + 1e-6 * (theta0 - theta_init)
                th_try = _apply_locks_bounds(th_try)
                th_new, ok = refit(th_try, x_all, y_b)
            except Exception as e:
                if err_msg is None:
                    err_msg = f"{type(e).__name__}: {e}"
            else:
                if ok and bool(fit.get("relabel_by_center", True)):
                    th_new = _relabel_by_center(np.asarray(th_new, float), theta0, block=4)
                    th_new = _apply_locks_bounds(th_new)
        elif not ok and use_linearized_fast_path:
            theta_lin = theta0.copy()
            step = None
            try:
                rhs = np.asarray(r_b, float).reshape(-1)
                step_free, *_ = np.linalg.lstsq(Jf, rhs, rcond=None)
                step = np.zeros_like(theta_lin)
                step[free_mask] = step_free
            except Exception as e:
                if err_msg is None:
                    err_msg = f"{type(e).__name__}: {e}"
            if step is not None:
                theta_lin = theta_lin + step
                theta_lin = _apply_locks_bounds(theta_lin)
                linear_fallbacks += 1
                linear_lambda = 0.0 if linear_lambda is None else linear_lambda
                if bool(fit.get("relabel_by_center", True)):
                    theta_lin = _relabel_by_center(np.asarray(theta_lin, float), theta0, block=4)
                    theta_lin = _apply_locks_bounds(theta_lin)
                th_new = np.asarray(theta_lin, float)
                ok = np.all(np.isfinite(th_new))
                if not strict_refit:
                    try:
                        theta_guess = _apply_locks_bounds(theta_lin)
                        th_refit, ok_refit = refit(theta_guess, x_all, y_b)
                    except Exception as e:
                        if err_msg is None:
                            err_msg = f"{type(e).__name__}: {e}"
                    else:
                        if ok_refit and bool(fit.get("relabel_by_center", True)):
                            th_refit = _relabel_by_center(np.asarray(th_refit, float), theta0, block=4)
                            th_refit = _apply_locks_bounds(th_refit)
                        th_new = _apply_locks_bounds(th_refit)
                        ok = bool(ok_refit and np.all(np.isfinite(th_new)))

        diag_failure["linear_fallbacks"] = int(linear_fallbacks)
        diag_failure["linear_lambda_last"] = (
            float(linear_lambda) if linear_lambda is not None else None
        )
        th_new = _apply_locks_bounds(th_new)

        return b, np.asarray(th_new, float), bool(ok), err_msg, (
            jitter_applied,
            jitter_reason,
            jitter_rms_local,
        )

    def _pulse(done_i: int):
        nonlocal next_pulse_at, last_pulse_t
        if progress_cb is not None and (done_i >= next_pulse_at or (time.monotonic() - last_pulse_t) > 0.5):
            try:
                progress_cb(f"Bootstrap: {done_i}/{int(n_boot)}")
            except Exception:
                pass
            last_pulse_t = time.monotonic()
            next_pulse_at = done_i + pulse_step

    jitter_applied_last = False
    jitter_reason_last = None
    jitter_rms_last = 0.0

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
                try:
                    _, th_new, ok, err_msg, jit_info = f.result()
                except Exception as e:
                    # Count as failed draw; capture first few error messages
                    n_fail += 1
                    _record_refit_error(f"{type(e).__name__}: {e}")
                    done_cnt += 1
                    _pulse(done_cnt)
                    continue

                jitter_applied_last, jitter_reason_last, jitter_rms_last = jit_info
                done_cnt += 1
                _pulse(done_cnt)
                if ok and np.all(np.isfinite(th_new)):
                    T_list.append(th_new)
                    n_success += 1
                else:
                    n_fail += 1
                    _record_refit_error(err_msg)
    else:
        for b in range(int(n_boot)):
            if abort_evt is not None:
                try:
                    if abort_evt.is_set():
                        aborted = True
                        break
                except Exception:
                    pass
            _, th_new, ok, err_msg, jit_info = _one_draw(int(b))
            jitter_applied_last, jitter_reason_last, jitter_rms_last = jit_info
            if ok and np.all(np.isfinite(th_new)):
                T_list.append(th_new)
                n_success += 1
            else:
                n_fail += 1
                _record_refit_error(err_msg)
            _pulse(b + 1)

    if not T_list or len(T_list) < 2:
        diag_failure_local = dict(diag_perf)
        diag_failure_local.update(diag_failure)
        diag_failure_local.update({
            "B": int(n_boot),
            "n_boot": int(n_boot),
            "n_success": int(n_success),
            "n_fail": int(n_fail),
            "n_linear_fallback": int(linear_fallbacks),
            "linear_lambda": float(linear_lambda) if linear_lambda is not None else None,
            "seed": seed,
            "bounds_source": bounds_source,
            "bounds_len": int(P),
            "bounds_any_finite": bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi))),
            "bounds_provided": bool(bounds_provided),
            "bounds_valid_shape": bool(bounds_valid_shape),
            "bounds_unpack_ok": bool(bounds_unpack_ok),
            "pct_at_bounds": None,
            "pct_at_bounds_units": "percent",
            "aborted": bool(aborted or ((abort_evt.is_set()) if abort_evt is not None else False)),
            "refit_errors": list(refit_errors[:16]),
            "alpha": float(alpha),
        })
        diag_failure_local["bootstrap_mode"] = "linearized" if use_linearized_fast_path else "refit"
        diag_failure_local.setdefault("refit_errors", [])
        err = RuntimeError(
            f"Insufficient successful bootstrap refits (success={n_success}, fail={n_fail})"
        )
        setattr(err, "diagnostics", diag_failure_local)
        raise err

    T = np.vstack(T_list)
    mean = T.mean(axis=0)
    sd = T.std(axis=0, ddof=1)
    qlo = np.quantile(T, alpha/2, axis=0)
    qhi = np.quantile(T, 1 - alpha/2, axis=0)

    names = param_names or [f"p{i}" for i in range(P)]
    stats = {names[i]: {"est": float(mean[i]), "sd": float(sd[i]), "p2.5": float(qlo[i]), "p97.5": float(qhi[i])}
             for i in range(P)}

    # % of successful thetas hitting bounds (diagnose degeneracy)
    pct_at_bounds = 0.0
    try:
        lo_hit = np.any(np.isclose(T, lo, rtol=0, atol=0), axis=1)
        hi_hit = np.any(np.isclose(T, hi, rtol=0, atol=0), axis=1)
        pct_at_bounds = float(100.0 * np.mean(lo_hit | hi_hit))
    except Exception:
        pct_at_bounds = 0.0
    # pct_at_bounds is expressed in percent (0-100)

    # Optional percentile band (subsample to control memory)
    band = None
    band_reason = None
    band_skip_reason = None
    band_gated = False
    diagnostics: Dict[str, object] = {}
    # Ensure band arrays are always defined even if band is gated/aborted
    band_lo: np.ndarray | None = None
    band_hi: np.ndarray | None = None

    # --- Safe defaults for band workers & perf diag ---
    try:
        _existing_bw_effective = band_workers_effective  # type: ignore[name-defined]
    except NameError:
        _existing_bw_effective = None
    if not _existing_bw_effective:
        try:
            bw_from_ctx = int((fit_ctx or {}).get("unc_band_workers", 0))
        except Exception:
            bw_from_ctx = 0
        try:
            workers_int = int(workers or 0)
        except Exception:
            workers_int = 0
        band_workers_effective = bw_from_ctx or workers_int
    else:
        band_workers_effective = _existing_bw_effective

    try:
        _ = diag_perf  # type: ignore[name-defined]
    except NameError:
        diag_perf = {}

    # Provide a local variable always present for the threadpool section
    try:
        band_workers  # type: ignore[name-defined]
    except NameError:
        band_workers = int(band_workers_effective or 0)
    if return_band:
        if predict_full is None or len(T_list) < BOOT_BAND_MIN_SAMPLES:
            band_reason = "missing model or insufficient samples"
            band_gated = True
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

            n_used = int(sel.size)
            if progress_cb:
                try:
                    progress_cb(
                        f"Bootstrap band: using {n_used} resamples, workers={int(band_workers_effective or 0)}"
                    )
                except Exception:
                    pass

            def _abort_requested() -> bool:
                return bool(abort_evt is not None and getattr(abort_evt, "is_set", lambda: False)())

            Y_list: List[np.ndarray] = []
            try:
                if band_workers and band_workers > 0:
                    with ThreadPoolExecutor(max_workers=band_workers) as ex:
                        for row_idx, pred_row in enumerate(ex.map(lambda idx: _eval_one(int(idx)), sel)):
                            if _abort_requested():
                                raise RuntimeError("aborted")
                            Y_list.append(pred_row)
                else:
                    for idx_val in sel:
                        if _abort_requested():
                            raise RuntimeError("aborted")
                        Y_list.append(_eval_one(int(idx_val)))
            except Exception as exc:
                aborted_eval = isinstance(exc, RuntimeError) and str(exc).lower() == "aborted"
                if progress_cb:
                    try:
                        progress_cb(
                            "Bootstrap band aborted." if aborted_eval else "Bootstrap band evaluation error."
                        )
                    except Exception:
                        pass
                diag_notes.append(repr(exc))
                band_gated = True
                band_skip_reason = BAND_SKIP_ABORTED if aborted_eval else "evaluation_error"
                band_reason = "aborted" if aborted_eval else "evaluation_error"
                Y_list = []
            else:
                band_backend = "numpy"
                try:
                    import cupy as cp  # optional
                    if use_gpu:
                        # If any element already on GPU, keep it; else copy to GPU
                        if any("cupy" in str(type(y)).lower() for y in Y_list):
                            Y = cp.stack([y if isinstance(y, cp.ndarray) else cp.asarray(y) for y in Y_list], axis=0)
                        else:
                            Y = cp.asarray(np.stack(Y_list, axis=0))
                        band_lo = cp.quantile(Y, float(alpha/2), axis=0)
                        band_hi = cp.quantile(Y, float(1 - alpha/2), axis=0)
                        band_lo = cp.asnumpy(band_lo)
                        band_hi = cp.asnumpy(band_hi)
                        band_backend = "cupy"
                    else:
                        Y = np.stack(Y_list, axis=0)
                        band_lo = np.quantile(Y, alpha/2, axis=0)
                        band_hi = np.quantile(Y, 1 - alpha/2, axis=0)
                except Exception:
                    if not Y_list:
                        raise RuntimeError("bootstrap_band_empty")
                    Y = np.stack(Y_list, axis=0)
                    band_lo = np.quantile(Y, alpha/2, axis=0)
                    band_hi = np.quantile(Y, 1 - alpha/2, axis=0)
                    band_backend = "numpy"

                # Build band tuple safely, even if x_all is None or lengths differ
                if band_lo is not None and band_hi is not None:
                    try:
                        n = int(band_lo.shape[0])
                    except Exception:
                        n = None

                    def _len(obj):
                        try:
                            return len(obj)
                        except Exception:
                            return None

                    need_default_x = (
                        x_all is None
                        or (n is not None and _len(x_all) != n)
                    )
                    if need_default_x:
                        x_all = np.arange(int(n or 0))
                    band = (x_all, band_lo, band_hi)
                    band_gated = False
                    band_skip_reason = None
                    band_reason = None
                else:
                    band = None
                    band_gated = True
                    band_skip_reason = "band_arrays_missing"
                    band_reason = band_reason or "band_arrays_missing"
                diagnostics["band_backend"] = band_backend
                try:
                    import cupy as cp  # optional
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

    diag = dict(diag_perf)
    diag.update({
        "B": int(n_boot),
        "n_boot": int(n_boot),
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "n_linear_fallback": int(linear_fallbacks),
        "linear_lambda": float(linear_lambda) if linear_lambda is not None else None,
        "seed": seed,
        "bounds_source": bounds_source,
        "bounds_len": int(P),
        "bounds_any_finite": bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi))),
        "bounds_provided": bool(bounds_provided),
        "bounds_valid_shape": bool(bounds_valid_shape),
        "bounds_unpack_ok": bool(bounds_unpack_ok),
        "pct_at_bounds": pct_at_bounds,
        "pct_at_bounds_units": "percent",
        "aborted": bool(aborted or ((abort_evt.is_set()) if abort_evt is not None else False)),
        "runtime_s": float(time.time() - t0),
        "band_source": "bootstrap-percentile" if band is not None else None,
        "band_reason": band_reason,
        "band_gated": bool(band_gated),
        "band_skip_reason": band_skip_reason,
        "band_workers_effective": (
            int(band_workers_effective)
            if band_workers_effective
            else int(band_workers or 0)
        ),
        "refit_errors": list(refit_errors[:16]),
        "alpha": float(alpha),
    })
    if diag_notes:
        diag["notes"] = diag_notes
    # Merge any per-band diagnostics (e.g. workers_used, band_backend)
    diag.update(diagnostics)
    diag.setdefault(
        "bounds_source",
        bounds_source if "bounds_source" in locals() else "unknown",
    )
    _bounds_any = False
    if "lo" in locals() and "hi" in locals():
        try:
            _bounds_any = bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))
        except Exception:
            _bounds_any = False
    diag.setdefault("bounds_any_finite", _bounds_any)
    diag.setdefault("bounds_len", int(P) if "P" in locals() else None)
    diag["bootstrap_mode"] = "linearized" if use_linearized_fast_path else "refit"
    diag["draw_workers_used"] = int(draw_workers) if draw_workers > 0 else None
    diag["band_workers_requested"] = (
        int(band_workers_requested) if band_workers_requested > 0 else None
    )
    if diag.get("band_workers_effective") is None:
        diag["band_workers_effective"] = (
            int(band_workers_effective) if band_workers_effective is not None else None
        )
    diag["theta_jitter_scale"] = float(jitter_scale)
    diag["jitter_applied_any"] = bool(jitter_scale > 0)
    diag["jitter_free_params"] = int(np.sum(free_mask))
    diag["jitter_last_rms"] = float(jitter_rms_last)
    diag["jitter_last_applied"] = bool(jitter_applied_last)
    # Record the numeric backend for parity across methods.
    try:
        diag["numpy_backend"] = performance.which_backend()
    except Exception:
        diag["numpy_backend"] = "numpy"
    if jitter_reason_last:
        diag["jitter_reason"] = jitter_reason_last

    def _add_percentile_aliases_inplace(stats_map: Dict[str, Dict[str, Any]]) -> None:
        for _k, blk in list(stats_map.items()):
            if not isinstance(blk, dict):
                continue
            if "p2.5" in blk and "p2_5" not in blk:
                blk["p2_5"] = blk["p2.5"]
            if "p97.5" in blk and "p97_5" not in blk:
                blk["p97_5"] = blk["p97.5"]
            if "p2_5" in blk and "p2.5" not in blk:
                blk["p2.5"] = blk["p2_5"]
            if "p97_5" in blk and "p97.5" not in blk:
                blk["p97.5"] = blk["p97_5"]

    _add_percentile_aliases_inplace(stats)

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


# --- MCSE for quantiles (Bayesian diagnostics; dependency-free)
def _mcse_quantile(samples: np.ndarray, q: float, batches: int = 20) -> float:
    """
    Batch-replicate MCSE for a quantile (q in [0,1]).
    Returns NaN if insufficient draws.
    """
    try:
        s = np.asarray(samples).ravel()
        n = s.size
        if n == 0 or batches < 2 or n < batches:
            return float("nan")
        k = n // batches
        if k < 2:
            return float("nan")
        qs = []
        for i in range(batches):
            chunk = s[i * k : (i + 1) * k]
            if chunk.size:
                qs.append(np.quantile(chunk, q))
        if len(qs) < 2:
            return float("nan")
        return float(np.std(qs, ddof=1) / np.sqrt(len(qs)))
    except Exception:
        return float("nan")


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

    fit_ctx = dict(fit_ctx or {})
    fit = fit_ctx
    # Normalize solver & any sharing hints (shared helper)
    _solver_name, _share_fwhm, _share_eta = _norm_solver_and_sharing(fit, {})
    fit["solver"] = _solver_name
    fit["share_fwhm"] = bool(_share_fwhm)
    fit["share_eta"] = bool(_share_eta)
    # IMPORTANT: By default, do NOT tie parameters inside MCMC.
    # Only respect sharing if explicitly enabled by user.
    if not bool(fit.get("bayes_respect_sharing", False)):
        fit["share_fwhm"] = False
        fit["share_eta"] = False

    theta_hat = np.asarray(theta_hat, float)
    diag_notes: List[str] = []
    # Significance level for credible intervals (used in per-parameter quantiles)
    alpha = float(fit_ctx.get("alpha", 0.05))
    sampler_cfg = dict(fit_ctx.get("bayes_sampler_cfg", {}))
    if seed is not None:
        sampler_cfg.setdefault("seed", seed)
    seed = sampler_cfg.get("seed", seed)
    strategy = str(
        sampler_cfg.get(
            "perf_parallel_strategy",
            fit_ctx.get("perf_parallel_strategy", "outer"),
        )
    )
    try:
        blas_threads = int(
            sampler_cfg.get(
                "perf_blas_threads",
                fit_ctx.get("perf_blas_threads", 0),
            )
            or 0
        )
    except Exception:
        blas_threads = 0
    sampler_cfg["perf_parallel_strategy"] = strategy
    sampler_cfg["perf_blas_threads"] = blas_threads
    fit_ctx["bayes_sampler_cfg"] = sampler_cfg
    fit_ctx.setdefault("perf_parallel_strategy", strategy)
    fit_ctx.setdefault("perf_blas_threads", blas_threads)
    try:
        seed_int = int(seed) if seed is not None else None
    except Exception:
        seed_int = None

    # Pull perf config early and derive the effective seed we will use everywhere
    cfg_perf = performance.get_parallel_config()
    seed_effective = (
        seed_int if seed_int is not None else (cfg_perf.seed_value if cfg_perf.seed_all else None)
    )
    seed_meta = seed_effective if seed_effective is not None else seed
    diag_perf = {
        "parallel_strategy": strategy,
        "blas_threads": int(blas_threads),
        "blas_effective": (
            1 if strategy == "outer" else (int(blas_threads) if blas_threads > 0 else None)
        ),
    }
    # Emit breadcrumbs for logs/debugging (no behavior change)
    diag_perf["solver_used"] = str(fit_ctx.get("solver", "")).lower()
    diag_perf["share_fwhm"] = bool(fit_ctx.get("share_fwhm", False))
    diag_perf["share_eta"] = bool(fit_ctx.get("share_eta", False))
    # Optional tying inside MCMC (OFF by default unless 'bayes_respect_sharing' True)
    share_fwhm = bool(fit_ctx.get("share_fwhm", False))
    share_eta = bool(fit_ctx.get("share_eta", False))
    # Toggle diagnostics (ESS/R̂/MCSE); default off; honor legacy key for compatibility
    _diag_flag = fit_ctx.get("bayes_diagnostics", None)
    if _diag_flag is None:
        _diag_flag = fit_ctx.get("bayes_diag", False)
    diagnostics_enabled = bool(_diag_flag)
    fc = dict(fit_ctx)
    bayes_band_enabled: bool = bool(fc.get("bayes_band_enabled", False))
    bayes_band_force: bool = bool(fc.get("bayes_band_force", False))
    bayes_band_max_draws: int = int(fc.get("bayes_band_max_draws", 512) or 512)
    # Surface settings into diagnostics for UI/debug logging.
    diag_perf["bayes_band_max_draws"] = int(bayes_band_max_draws)
    thr_ess_min = float(fc.get("bayes_diag_ess_min", 200.0))
    thr_rhat_max = float(fc.get("bayes_diag_rhat_max", 1.05))
    thr_mcse_mean = float(fc.get("bayes_diag_mcse_mean", float("inf")))
    band_workers = fc.get("unc_band_workers") if fc else None
    band_workers_requested = 0
    try:
        band_workers_requested = int(band_workers) if band_workers is not None else 0
    except Exception:
        band_workers_requested = 0
    _cpu = os.cpu_count() or 1
    band_workers_effective = (
        max(1, min(band_workers_requested, _cpu)) if band_workers_requested > 0 else None
    )
    band_workers = (
        band_workers_effective
        if (band_workers_effective is not None and band_workers_effective > 1)
        else None
    )
    progress_cb = fc.get("progress_cb") if fc else None
    abort_evt = fc.get("abort_event") if fc else None
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
            lp += -0.5*(sigma/rmse)**2 + math.log(2.0) - math.log(max(sigma, 1e-300))
        else:
            s = rmse
            lp += math.log(2.0/ math.pi) - math.log(s*(1.0 + (sigma/s)**2)) - math.log(max(sigma, 1e-300))
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
    rng = np.random.default_rng(seed_effective)
    # Optional progress + abort
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
        val_workers = fit_ctx.get("unc_workers", 0)
    try:
        w = max(0, min(int(val_workers or 0), os.cpu_count() or 1))
    except Exception:
        w = 0
    pool = ThreadPoolExecutor(max_workers=w) if w > 0 else None
    sampler = None
    limit = 1 if strategy == "outer" else (blas_threads if blas_threads > 0 else None)
    performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
    try:
        with performance.blas_limit_ctx(limit):
            try:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    dim,
                    lambda z: log_prob(z),
                    pool=pool,
                    random_state=seed_effective if seed_effective is not None else None,
                )
            except TypeError:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    dim,
                    lambda z: log_prob(z),
                    pool=pool,
                )
                if seed_effective is not None:
                    try:
                        sampler.random_state = np.random.RandomState(int(seed_effective))
                    except Exception:
                        pass
            # Guardrails before sampling for very high dimension
            if dim > 40:
                n_burn = min(int(n_burn), 1000)
                n_steps = min(int(n_steps), 4000)

            draws = []
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
                performance.apply_global_seed(cfg_perf.seed_value, cfg_perf.seed_all)
                state, lnp, _ = sampler.run_mcmc(
                    state, step, progress=False, skip_initial_state_check=True
                )
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
        diag = dict(diag_perf)
        diag.update({"seed": seed_meta, "aborted": True, "n_draws": 0, "band_source": None})
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics=diag,
            band=None,
        )

    chain_post = sampler.get_chain(discard=n_burn, thin=thin, flat=False)  # (steps, walkers, dim)
    acc_frac = float(np.mean(sampler.acceptance_fraction))
    if chain_post.ndim != 3:
        chain_post = np.asarray(chain_post).reshape((-1, n_walkers, dim))
    n_post = chain_post.shape[0]  # number of post-burn/thinned steps per walker
    if n_post <= 0:
        # Avoid "index -1" crashes when burn==total or abort mid-burn
        diag = dict(diag_perf)
        diag.update(
            {
                "seed": seed_meta,
                "aborted": True,
                "n_draws": 0,
                "accept_frac_mean": acc_frac,
                "band_source": None,
            }
        )
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics=diag,
            band=None,
        )
    n_samp = n_post * n_walkers

    flat = chain_post.reshape(-1, dim)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    n_samp = flat.shape[0]
    if n_samp < 2:
        diag = dict(diag_perf)
        diag.update(
            {
                "seed": seed_meta,
                "aborted": True,
                "n_draws": 0,
                "accept_frac_mean": acc_frac,
                "band_source": None,
            }
        )
        return UncertaintyResult(
            method="bayesian",
            label="Bayesian (MCMC)",
            stats={},
            diagnostics=diag,
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
        return {
            "est": vals,
            "sd": sds,
            "p2.5": lo2,
            "p97.5": hi97,
            "p2_5": lo2,
            "p97_5": hi97,
        }

    param_stats: Dict[str, Any] = {
        "center": _stat_vec(0),
        "height": _stat_vec(1),
        "fwhm": _stat_vec(2),
        "eta": _stat_vec(3),
    }

    finite = np.isfinite(sigma_draws)
    if finite.any():
        sigma_fin = sigma_draws[finite]
        p25_sigma = float(np.nanpercentile(sigma_fin, 100.0 * (alpha / 2)))
        p975_sigma = float(np.nanpercentile(sigma_fin, 100.0 * (1.0 - alpha / 2)))
        param_stats["sigma"] = {
            "est": float(np.nanmedian(sigma_fin)),
            "sd": _nanstd_safe(sigma_fin),
            "p2.5": p25_sigma,
            "p97.5": p975_sigma,
            "p2_5": p25_sigma,
            "p97_5": p975_sigma,
        }
    else:
        nan = float("nan")
        param_stats["sigma"] = {
            "est": nan,
            "sd": nan,
            "p2.5": nan,
            "p97.5": nan,
            "p2_5": nan,
            "p97_5": nan,
        }

    def _scalar_stats_for(samples: np.ndarray) -> Dict[str, float]:
        finite = np.isfinite(samples)
        if finite.any():
            sf = samples[finite]
            est = float(np.nanmedian(sf))
            sd = float(_nanstd_safe(sf))
            p25 = float(np.nanpercentile(sf, 100.0 * (alpha / 2)))
            p975 = float(np.nanpercentile(sf, 100.0 * (1.0 - alpha / 2)))
            return {
                "est": est,
                "sd": sd,
                "p2.5": p25,
                "p97.5": p975,
                "p2_5": p25,
                "p97_5": p975,
            }
        nan = float("nan")
        return {
            "est": nan,
            "sd": nan,
            "p2.5": nan,
            "p97.5": nan,
            "p2_5": nan,
            "p97_5": nan,
        }

    default_names = [f"p{i}" for i in range(P)]
    custom_names = list(param_names) if param_names is not None else []
    for idx in range(P):
        draws_flat = T_full[:, idx]
        stats_flat = _scalar_stats_for(draws_flat)
        try:
            alpha_lo = float(alpha) / 2.0
            alpha_hi = 1.0 - alpha_lo
        except Exception:
            alpha_lo = 0.025
            alpha_hi = 0.975
        lo_mcse = _mcse_quantile(draws_flat, alpha_lo)
        hi_mcse = _mcse_quantile(draws_flat, alpha_hi)
        stats_flat["ci_lo_mcse"] = float(lo_mcse)
        stats_flat["ci_hi_mcse"] = float(hi_mcse)
        param_stats[default_names[idx]] = stats_flat
        if idx < len(custom_names):
            key = str(custom_names[idx])
            if key and key not in param_stats:
                param_stats[key] = dict(stats_flat)

    base_param_stats = dict(param_stats)

    ess_min = float("nan")
    rhat_max = float("nan")
    mcse_mean = float("nan")

    # (steps, walkers, free_params) view for diagnostics and band gating
    post = chain_post[:, :, :P_free] if P_free > 0 else None
    post_swapped = None
    try:
        # emcee returns (nsteps, nwalkers, nparams)
        chain_full = sampler.get_chain(flat=False)
        post_full = np.asarray(chain_full, float)
        steps_take = int(n_steps)
        if steps_take > 0 and post_full.shape[0] >= steps_take:
            post_full = post_full[-steps_take:, :, :]
        t = int(thin)
        if t > 1:
            post_full = post_full[::t, :, :]
        post = post_full[:, :, :P_free] if P_free > 0 else None
    except Exception as e:
        diag_notes.append(repr(e))

    if (
        diagnostics_enabled
        and post is not None
        and post.size
        and callable(progress_cb)
    ):
        try:
            progress_cb("Computing Bayesian diagnostics (ESS/R̂/MCSE)…")
        except Exception:
            pass

    if diagnostics_enabled and post is not None and post.size:
        try:
            # diagnostics expect (nchains, ndraws, nparams)
            post_swapped = np.swapaxes(post, 0, 1)
            ess_val = ess_autocorr(post_swapped)
            ess_arr = np.asarray(ess_val, dtype=float)
            if ess_arr.size:
                ess_min = float(np.nanmin(ess_arr))
        except Exception as e:
            diag_notes.append(repr(e))
        try:
            post_for_rhat = post_swapped if post_swapped is not None else np.swapaxes(post, 0, 1)
            rhat_val = rhat_split(post_for_rhat)
            rhat_arr = np.asarray(rhat_val, dtype=float)
            if rhat_arr.size:
                rhat_max = float(np.nanmax(rhat_arr))
        except Exception as e:
            diag_notes.append(repr(e))
        try:
            if np.isfinite(ess_min) and ess_min > 0 and T_full.size:
                sd_vals = np.nanstd(T_full, axis=0, ddof=1)
                mcse_vals = np.asarray(sd_vals, float) / np.sqrt(max(ess_min, 1.0))
                if mcse_vals.size:
                    # Mean MCSE is more stable than the worst-parameter max.
                    mcse_mean = float(np.nanmean(mcse_vals))
        except Exception as e:
            diag_notes.append(repr(e))

    # --- Build per-peak stats (list of dicts) from draws, using param_names to map columns.
    try:
        if T_full.size and n_pk > 0:
            try:
                alpha_lo = float(alpha) / 2.0
                alpha_hi = 1.0 - alpha_lo
            except Exception:
                alpha_lo, alpha_hi = 0.025, 0.975

            # Build column index map from param_names if provided
            name_to_idx: Dict[str, int] = {}
            if isinstance(param_names, (list, tuple)):
                for idx, nm in enumerate(param_names):
                    try:
                        name_to_idx[str(nm)] = int(idx)
                    except Exception:
                        pass

            def _find_col(pk_idx: int, kind: str, fallback: int) -> int:
                """Prefer param_names mapping; fallback to 4*i+offset if not found."""
                if name_to_idx:
                    cand = [
                        f"{kind}{pk_idx+1}",
                        f"{kind}_{pk_idx+1}",
                        f"{kind}[{pk_idx+1}]",
                    ]
                    for c in cand:
                        if c in name_to_idx:
                            return int(name_to_idx[c])
                return int(fallback)

            def _one_param_stats(arr: np.ndarray) -> Dict[str, Any]:
                # Base scalar stats
                s = _scalar_stats_for(arr)
                # Force scalars for est/sd
                try:
                    s["est"] = float(s.get("est", np.nan))
                except Exception:
                    s["est"] = float("nan")
                try:
                    s["sd"] = float(s.get("sd", np.nan))
                except Exception:
                    s["sd"] = float("nan")
                # Ensure both dotted and underscored aliases exist
                p_lo = s.get("p2_5", s.get("p2.5", None))
                p_hi = s.get("p97_5", s.get("p97.5", None))
                if p_lo is not None:
                    s.setdefault("p2_5", p_lo)
                    s.setdefault("p2.5", p_lo)
                if p_hi is not None:
                    s.setdefault("p97_5", p_hi)
                    s.setdefault("p97.5", p_hi)
                # MCSE on CI endpoints
                try:
                    s["ci_lo_mcse"] = float(_mcse_quantile(arr, alpha_lo))
                    s["ci_hi_mcse"] = float(_mcse_quantile(arr, alpha_hi))
                except Exception:
                    pass
                return s

            stats_by_peak = []
            for pk_idx in range(int(n_pk)):
                base = 4 * pk_idx
                i_center = _find_col(pk_idx, "center", base + 0)
                i_height = _find_col(pk_idx, "height", base + 1)
                i_fwhm = _find_col(pk_idx, "fwhm", base + 2)
                i_eta = _find_col(pk_idx, "eta", base + 3)
                # Guard against out-of-range indices
                cols = T_full.shape[1]

                def _safe_col(i: int) -> np.ndarray:
                    return T_full[:, min(max(0, i), cols - 1)] if cols else np.asarray([])

                row = {
                    "center": _one_param_stats(_safe_col(i_center)),
                    "height": _one_param_stats(_safe_col(i_height)),
                    "fwhm": _one_param_stats(_safe_col(i_fwhm)),
                    "eta": _one_param_stats(_safe_col(i_eta)),
                }
                # If a field ended with NaN, backfill from vector blocks (same index) when present.
                for _k in ("center", "height", "fwhm", "eta"):
                    blk = base_param_stats.get(_k) if isinstance(base_param_stats, dict) else None
                    if isinstance(blk, dict):
                        try:
                            est_list = blk.get("est")
                            sd_list = blk.get("sd")
                            if not np.isfinite(row[_k].get("est", np.nan)) and isinstance(est_list, (list, tuple, np.ndarray)):
                                row[_k]["est"] = float(est_list[pk_idx]) if pk_idx < len(est_list) else float("nan")
                            if not np.isfinite(row[_k].get("sd", np.nan)) and isinstance(sd_list, (list, tuple, np.ndarray)):
                                row[_k]["sd"] = float(sd_list[pk_idx]) if pk_idx < len(sd_list) else float("nan")
                        except Exception:
                            pass
                stats_by_peak.append(row)
            if stats_by_peak:
                # Preserve any existing vector blocks and append per-peak rows for UI consumption.
                new_param_stats: Dict[str, Any] = dict(base_param_stats)
                new_param_stats["rows"] = stats_by_peak
                param_stats = new_param_stats
    except Exception as exc:
        diag_notes.append(repr(exc))

    # --- Decide band computation based on diagnostics (disable unless healthy or explicitly forced)
    try:
        band_pref = bool((fit_ctx or {}).get("bayes_band_enabled", False)) and bool(return_band)
        band_force = bool((fit_ctx or {}).get("bayes_band_force", False))
        ess_thr = float((fit_ctx or {}).get("bayes_band_ess_min", 200.0))
        rhat_thr = float((fit_ctx or {}).get("bayes_band_rhat_max", 1.2))
        diag_good = (np.isfinite(ess_min) and ess_min >= ess_thr) and (not np.isfinite(rhat_max) or rhat_max <= rhat_thr)
        if band_pref and not (diag_good or band_force):
            return_band = False
            diag_notes.append("bayes_band_skipped: diagnostics failed and force disabled")
        elif band_pref and band_force and not diag_good:
            diag_notes.append("bayes_band_forced: diagnostics failed")
    except Exception:
        pass

    # --- Retain CI MCSE diagnostics (unchanged contract)
    ci_mcse_diag = {}
    try:
        if T_full.size:
            alpha_lo = float(alpha) / 2.0
            alpha_hi = 1.0 - alpha_lo
        else:
            alpha_lo = float(alpha) / 2.0
            alpha_hi = 1.0 - alpha_lo
    except Exception:
        alpha_lo = 0.025
        alpha_hi = 0.975
    try:
        if T_full.size:
            diag_ci: Dict[str, Any] = {}
            if n_pk > 0:
                for name, offset in (("center", 0), ("height", 1), ("fwhm", 2), ("eta", 3)):
                    blk = param_stats if isinstance(param_stats, dict) else None
                    lo_list: List[float] = []
                    hi_list: List[float] = []
                    for i in range(int(n_pk)):
                        draws_i = T_full[:, 4 * i + offset]
                        lo_list.append(float(_mcse_quantile(draws_i, alpha_lo)))
                        hi_list.append(float(_mcse_quantile(draws_i, alpha_hi)))
                    if blk is not None:
                        blk_stats = blk.get(name)
                        if isinstance(blk_stats, dict):
                            blk_stats["ci_lo_mcse"] = float(np.nanmean(lo_list)) if lo_list else float("nan")
                            blk_stats["ci_hi_mcse"] = float(np.nanmean(hi_list)) if hi_list else float("nan")
                    diag_ci[name] = {
                        "ci_lo_mcse": float(np.nanmean(lo_list)) if lo_list else float("nan"),
                        "ci_hi_mcse": float(np.nanmean(hi_list)) if hi_list else float("nan"),
                    }
            sig_blk = param_stats.get("sigma") if isinstance(param_stats, dict) else None
            # Guard: if sigma draws were not captured in this code path, skip MCSE for σ gracefully.
            try:
                _sig_draws = sigma_draws  # noqa: F821
            except NameError:
                _sig_draws = None
            if isinstance(sig_blk, dict) and _sig_draws is not None:
                lo_mcse = float(_mcse_quantile(_sig_draws, alpha_lo))
                hi_mcse = float(_mcse_quantile(_sig_draws, alpha_hi))
                sig_blk["ci_lo_mcse"] = lo_mcse
                sig_blk["ci_hi_mcse"] = hi_mcse
                diag_ci["sigma"] = {"ci_lo_mcse": lo_mcse, "ci_hi_mcse": hi_mcse}
            if diag_ci:
                ci_mcse_diag = diag_ci
    except Exception:
        ci_mcse_diag = {}

    # --- OPTIONAL: summarize CI MCSE for logs (populated later if available)
    ci_mcse_diag_for_log: Dict[str, Any] = dict(ci_mcse_diag)

    diag = dict(diag_perf)
    diag.update({
        "n_draws": int(n_samp),
        "ess_min": ess_min,
        "rhat_max": rhat_max,
        "mcse_mean": mcse_mean,
        "accept_frac_mean": acc_frac,
        "seed": seed_meta,
        "aborted": bool(aborted),
        "band_source": None,
        "band_gated": False,
        "band_forced": False,
        "diagnostics_enabled": bool(diagnostics_enabled),
        # Fill band worker diagnostics after band logic for consistency.
    })
    if ci_mcse_diag_for_log:
        diag["ci_mcse"] = ci_mcse_diag_for_log
    # Record backend used for the Bayesian path too.
    try:
        diag["numpy_backend"] = performance.which_backend()
    except Exception:
        diag["numpy_backend"] = "numpy"
    if diag_notes:
        diag["notes"] = diag_notes

    band_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    should_attempt_band = bool(bayes_band_enabled) and bool(post is not None and post.size) and callable(predict_full)
    if should_attempt_band and not (diagnostics_enabled or bayes_band_force):
        if callable(progress_cb):
            try:
                progress_cb(
                    "Bayesian band skipped: diagnostics are OFF (turn on diagnostics or toggle 'Force band')."
                )
            except Exception:
                pass
        diag["band_gated"] = True
        diag["band_forced"] = False
        diag["band_skip_reason"] = BAND_SKIP_DIAG_DISABLED
        should_attempt_band = False

    if should_attempt_band:
        n_total = int(n_samp)
        max_draws = max(1, int(bayes_band_max_draws))
        min_required = min(BAYES_BAND_MIN_DRAWS, max_draws)
        # Expose the computed minimum requirement.
        diag["bayes_band_min_required"] = int(min_required)
        if n_total < min_required:
            diag["band_gated"] = True
            diag["band_forced"] = bool(bayes_band_force)
            diag["band_draws_used"] = int(n_total)
            diag["band_skip_reason"] = BAND_SKIP_INSUFF_DRAWS
            if callable(progress_cb):
                try:
                    progress_cb(
                        f"Bayesian band skipped: draws={n_total} < required {min_required}"
                    )
                except Exception:
                    pass
        else:
            # Clamp BLAS for the prediction pass (parity with bootstrap band path)
            with performance.blas_limit_ctx(limit):
                stride = 1 if n_total <= max_draws else int(np.ceil(n_total / max_draws))
                idx = np.arange(0, n_total, stride, dtype=int)
                if idx.size > max_draws:
                    idx = idx[:max_draws]
                n_used = int(idx.size)

                if callable(progress_cb):
                    try:
                        progress_cb(
                            f"Bayesian band: draws={n_total}, using {n_used} (stride={stride}), "
                            f"workers={band_workers_effective or 0}"
                        )
                    except Exception:
                        pass

                hard_failure = bool(aborted) or n_used <= 0
                gate_bad = False
                if diagnostics_enabled and not bayes_band_force:
                    cond_ess = (not np.isfinite(thr_ess_min)) or (np.isfinite(ess_min) and ess_min >= thr_ess_min)
                    cond_rhat = (not np.isfinite(thr_rhat_max)) or (np.isfinite(rhat_max) and rhat_max <= thr_rhat_max)
                    cond_mcse = (not np.isfinite(thr_mcse_mean)) or (np.isfinite(mcse_mean) and mcse_mean <= thr_mcse_mean)
                    gate_bad = (not cond_ess) or (not cond_rhat) or (not cond_mcse) or bool(aborted)

                if hard_failure:
                    diag["band_gated"] = True
                    diag["band_forced"] = bool(bayes_band_force)
                    diag["band_draws_used"] = n_used
                    diag["band_skip_reason"] = BAND_SKIP_HARD_FAILURE
                elif gate_bad and not bayes_band_force:
                    diag["band_gated"] = True
                    diag["band_forced"] = False
                    diag["band_draws_used"] = n_used
                    diag["band_skip_reason"] = BAND_SKIP_DIAG_UNHEALTHY
                    diag["band_gating_thresholds"] = {
                        "ess_min_req": thr_ess_min,
                        "ess_min": ess_min,
                        "rhat_max_req": thr_rhat_max,
                        "rhat_max": rhat_max,
                        "mcse_mean_req": thr_mcse_mean,
                        "mcse_mean": mcse_mean,
                    }
                    if callable(progress_cb):
                        try:
                            progress_cb(
                                f"Bayesian band gated: ESS_min={ess_min:.3g} (≥{thr_ess_min}), "
                                f"R̂_max={rhat_max:.3g} (≤{thr_rhat_max}), MCSE_mean={mcse_mean:.3g} (≤{thr_mcse_mean})."
                            )
                        except Exception:
                            pass
                else:
                    diag["band_gated"] = False
                    diag["band_forced"] = bool(bayes_band_force)
                    diag["band_draws_used"] = n_used
                    diag["band_stride"] = int(stride)
                    diag.pop("band_skip_reason", None)
                    diag.pop("band_gating_thresholds", None)
                    theta_sel = T_full[idx, :]

                    def _evaluate(theta_vec: np.ndarray) -> np.ndarray:
                        return np.asarray(predict_full(theta_vec), float)

                    xb = np.asarray(x_all, float)
                    Y = np.empty((n_used, xb.size), float)
                    try:
                        if band_workers and band_workers > 1:
                            with ThreadPoolExecutor(max_workers=band_workers) as pool:
                                for row_idx, pred_row in enumerate(pool.map(_evaluate, theta_sel)):
                                    if abort_evt is not None and getattr(abort_evt, "is_set", lambda: False)():
                                        raise RuntimeError("aborted")
                                    Y[row_idx, :] = np.asarray(pred_row, float)
                        else:
                            for row_idx in range(n_used):
                                if abort_evt is not None and getattr(abort_evt, "is_set", lambda: False)():
                                    raise RuntimeError("aborted")
                                Y[row_idx, :] = np.asarray(_evaluate(theta_sel[row_idx, :]), float)
                    except Exception as exc:
                        diag_notes.append(repr(exc))
                        diag["band_gated"] = True
                        diag["band_forced"] = bool(bayes_band_force)
                        diag["band_skip_reason"] = (
                            BAND_SKIP_ABORTED
                            if isinstance(exc, RuntimeError) and str(exc).lower() == "aborted"
                            else "evaluation_error"
                        )
                    else:
                        qlo = 100.0 * (alpha / 2.0)
                        qhi = 100.0 * (1.0 - alpha / 2.0)
                        lo = np.nanpercentile(Y, qlo, axis=0)
                        hi = np.nanpercentile(Y, qhi, axis=0)
                        band_tuple = (xb, lo, hi)
                        diag["band_source"] = "posterior_predictive_subset"
                        diag["band_backend"] = "numpy"
                        if callable(progress_cb):
                            try:
                                if diag["band_forced"]:
                                    progress_cb("Bayesian band computed (FORCED despite diagnostics).")
                                else:
                                    progress_cb("Bayesian band computed.")
                            except Exception:
                                pass

    diag["band_workers_requested"] = (
        int(band_workers_requested) if band_workers_requested > 0 else None
    )
    diag["band_workers_effective"] = (
        int(band_workers_effective) if band_workers_effective is not None else None
    )

    if diag_notes and "notes" not in diag:
        diag["notes"] = diag_notes

    return UncertaintyResult(
        method="bayesian",
        label="Bayesian (MCMC)",
        stats=param_stats,
        diagnostics=diag,
        band=band_tuple,
    )
