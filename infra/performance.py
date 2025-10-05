"""Defensive performance layer with optional Numba/CuPy backends.

All public evaluators return ``np.ndarray`` with ``dtype=float64``.  The
selected backend is one of ``{"numpy", "numba", "cupy"}`` and can be
switched at runtime.  When shadow-compare is enabled (either explicitly or
via ``GL_PERF_SHADOW_COMPARE``), the fast path is cross-checked against the
NumPy reference; on any mismatch a single warning is logged and the backend
permanently falls back to NumPy for the remainder of the process.

The math mirrors the pseudo-Voigt formula used throughout the GUI and keeps
summation order deterministic by stacking components and using ``np.sum``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import random
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
try:  # pragma: no cover - optional import
    import numba as _numba
    _NUMBA_OK = True
except Exception:  # pragma: no cover - numba not available
    _numba = None
    _NUMBA_OK = False

try:  # pragma: no cover - optional import
    import cupy as _cp
    _CUPY_OK = True
except Exception:  # pragma: no cover - cupy not available
    _cp = None
    _CUPY_OK = False

try:  # pragma: no cover - optional import
    from threadpoolctl import ThreadpoolController
except Exception:  # pragma: no cover - threadpoolctl not available
    ThreadpoolController = None


# ---------------------------------------------------------------------------
# Global state controlled via setters
_NUMBA_USER = False
_GPU_USER = False
_BACKEND = "numpy"

_CACHE_BASELINE = True
_MAX_WORKERS = 0
_UNC_WORKERS = 0
_SEED: Optional[int] = None
_SEED_ALL = False
_GPU_CHUNK = 262_144
_TPCTL_NOTED = False

_LOG: Optional[Callable[[str, str], None]] = None

_SHADOW_COMPARE = False
_SHADOW_RTOL = 1e-10
_SHADOW_ATOL = 1e-12
_SHADOW_WARNED = False


# ---------------------------------------------------------------------------
# Logger helpers
def set_logger(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register a logger callback ``fn(message, level)``."""

    global _LOG
    _LOG = fn


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG is None:
        return
    try:  # pragma: no cover - defensive
        _LOG(msg, level)
    except Exception:
        pass


def note_tpctl_absence_once() -> None:
    """Log a single debug note when threadpoolctl is unavailable."""

    global _TPCTL_NOTED
    if not _TPCTL_NOTED:
        try:
            _log(
                "threadpoolctl not available; BLAS clamp relies on env vars only.",
                "DEBUG",
            )
        except Exception:
            pass
        _TPCTL_NOTED = True


# ---------------------------------------------------------------------------
# Backend management
def _update_backend() -> None:
    global _BACKEND
    if _GPU_USER and _CUPY_OK:
        _BACKEND = "cupy"
    elif _NUMBA_USER and _NUMBA_OK:
        _BACKEND = "numba"
    else:
        _BACKEND = "numpy"


def set_numba(enabled: bool) -> None:
    """Enable Numba JIT backend when available."""

    global _NUMBA_USER
    _NUMBA_USER = bool(enabled) and _NUMBA_OK
    if enabled and not _NUMBA_OK:
        _log("Numba not available; running on NumPy.", "WARN")
    if _NUMBA_USER:
        _warmup_numba()
    _update_backend()


def set_gpu(enabled: bool) -> None:
    """Enable CuPy GPU backend when available."""

    global _GPU_USER
    _GPU_USER = bool(enabled) and _CUPY_OK
    if enabled and not _CUPY_OK:
        _log("CuPy not available; GPU disabled.", "WARN")
    if _GPU_USER:
        try:  # pragma: no cover - GPU initialisation
            _warmup_gpu()
        except Exception:
            _GPU_USER = False
            _log("CuPy GPU init failed; falling back to NumPy.", "WARN")
    _update_backend()


def set_cache_baseline(enabled: bool) -> None:
    global _CACHE_BASELINE
    _CACHE_BASELINE = bool(enabled)


def cache_baseline_enabled() -> bool:
    return _CACHE_BASELINE


def set_seed(seed: Optional[int]) -> None:
    """Record a preferred seed for deterministic ops without globally seeding NumPy."""

    global _SEED
    _SEED = int(seed) if seed is not None else None
    # Do NOT call np.random.seed() here; callers (bootstrap/emcee/etc.) pass explicit seeds for reproducibility.
    try:
        import cupy as _cp  # type: ignore
        _ = _cp.random  # ensure module loads
        # Avoid global CuPy seeding here as well
    except Exception:
        pass


def set_max_workers(n: int) -> None:
    global _MAX_WORKERS
    _MAX_WORKERS = max(0, int(n))


def set_unc_workers(n: int) -> None:
    """Set uncertainty worker cap (0 ⇒ auto)."""

    global _UNC_WORKERS
    _UNC_WORKERS = max(0, int(n))


def get_unc_workers() -> int:
    return _UNC_WORKERS


def get_max_workers() -> int:
    return _MAX_WORKERS


def _resolve_workers(n: Optional[int]) -> int:
    if not n:
        try:
            return max(1, os.cpu_count() or 1)
        except Exception:
            return 1
    try:
        return max(1, int(n))
    except Exception:
        return 1


def apply_global_seed(seed: Optional[int], enable: bool) -> None:
    """Apply deterministic seeding across supported RNGs."""

    global _SEED, _SEED_ALL
    _SEED_ALL = bool(enable)
    _SEED = int(seed) if seed is not None else None
    if not _SEED_ALL or _SEED is None:
        return

    try:
        random.seed(_SEED)
    except Exception:
        pass

    try:
        np.random.seed(_SEED)
    except Exception:
        pass

    if _cp is not None:
        try:  # pragma: no cover - optional backend
            _cp.random.seed(_SEED)
        except Exception:
            pass


@contextmanager
def blas_single_thread_ctx():
    """Limit BLAS/OpenMP libraries to a single thread within the context."""

    env_keys = ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    prev_env = {k: os.environ.get(k) for k in env_keys}
    controller = None
    try:
        if ThreadpoolController is not None:  # pragma: no branch - optional
            try:
                controller = ThreadpoolController()
                controller.limit(limits=1)
            except Exception:
                controller = None
        else:
            note_tpctl_absence_once()
        for key in env_keys:
            os.environ[key] = "1"
        yield
    finally:
        if controller is not None:
            try:  # pragma: no cover - optional backend restore
                controller.restore_initial_limits()
            except Exception:
                pass
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def blas_limit_ctx(threads: int | None):
    """Limit BLAS/OpenMP threadpools within the scope.

    ``None`` or ``<=0`` ⇒ no clamp (leave libraries as-is).
    """

    try:
        if threads is None or int(threads) <= 0:
            yield
            return
        threads = int(threads)
    except Exception:
        yield
        return

    env_keys = ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    prev_env = {k: os.environ.get(k) for k in env_keys}
    controller = None
    try:
        if ThreadpoolController is not None:  # pragma: no branch - optional
            try:
                controller = ThreadpoolController()
                controller.limit(limits=threads)
            except Exception:
                controller = None
        else:
            note_tpctl_absence_once()

        for key in env_keys:
            os.environ[key] = str(threads)
        yield
    finally:
        if controller is not None:
            try:  # pragma: no cover - optional backend restore
                controller.restore_initial_limits()
            except Exception:
                pass
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class ParallelConfig:
    fit_workers: int
    unc_workers: int
    seed_all: bool = False
    seed_value: Optional[int] = None


def get_parallel_config() -> ParallelConfig:
    return ParallelConfig(
        fit_workers=_resolve_workers(_MAX_WORKERS),
        unc_workers=_resolve_workers(_UNC_WORKERS),
        seed_all=_SEED_ALL,
        seed_value=_SEED,
    )


def set_gpu_chunk(n: int) -> None:
    global _GPU_CHUNK
    _GPU_CHUNK = max(16_384, int(n))


def which_backend() -> str:
    return _BACKEND


def enable_shadow_compare(flag: bool, rtol: float = 1e-10, atol: float = 1e-12) -> None:
    global _SHADOW_COMPARE, _SHADOW_RTOL, _SHADOW_ATOL
    _SHADOW_COMPARE = bool(flag)
    _SHADOW_RTOL = float(rtol)
    _SHADOW_ATOL = float(atol)


def _shadow_fail() -> None:
    """Handle shadow-compare mismatch by logging and falling back to NumPy."""

    global _SHADOW_WARNED, _NUMBA_USER, _GPU_USER
    if not _SHADOW_WARNED:
        _log("Performance backend mismatch; falling back to NumPy.", "WARN")
        _SHADOW_WARNED = True
    _NUMBA_USER = False
    _GPU_USER = False
    _update_backend()


# Enable shadow compare if environment variable is set ----------------------
if os.getenv("GL_PERF_SHADOW_COMPARE", "").strip():  # pragma: no cover - env hook
    enable_shadow_compare(True)


# ---------------------------------------------------------------------------
# Core math kernels
def _pvoigt_np(x, h, c, w, eta):
    w = 1e-12 if w < 1e-12 else w
    eta = 0.0 if eta < 0.0 else 1.0 if eta > 1.0 else eta
    dx = (x - c) / w
    ga = np.exp(-4.0 * np.log(2.0) * dx * dx)
    lo = 1.0 / (1.0 + 4.0 * dx * dx)
    return h * ((1.0 - eta) * ga + eta * lo)


if _NUMBA_OK:  # pragma: no cover - JIT definitions
    @_numba.njit(cache=True)
    def _pvoigt_nb(x, h, c, w, eta):  # type: ignore
        w = 1e-12 if w < 1e-12 else w
        if eta < 0.0:
            eta = 0.0
        elif eta > 1.0:
            eta = 1.0
        dx = (x - c) / w
        ga = np.exp(-4.0 * np.log(2.0) * dx * dx)
        lo = 1.0 / (1.0 + 4.0 * dx * dx)
        return h * ((1.0 - eta) * ga + eta * lo)


def _pvoigt_cp(x_cp, h, c, w, eta):  # pragma: no cover - GPU path
    w = 1e-12 if w < 1e-12 else w
    eta = 0.0 if eta < 0.0 else 1.0 if eta > 1.0 else eta
    dx = (x_cp - c) / w
    ga = _cp.exp(-4.0 * math.log(2.0) * dx * dx)
    lo = 1.0 / (1.0 + 4.0 * dx * dx)
    return h * ((1.0 - eta) * ga + eta * lo)


# ---------------------------------------------------------------------------
# Backend-specific evaluators
def _eval_components_numpy(x, peaks):
    comps = [
        _pvoigt_np(x, h, c, w, eta) for (h, c, w, eta) in peaks
    ]
    if comps:
        return np.vstack(comps).astype(np.float64, copy=False)
    return np.zeros((0, x.size), dtype=np.float64)


def _eval_total_numpy(x, peaks):
    comps = _eval_components_numpy(x, peaks)
    if comps.size:
        return np.sum(comps, axis=0, dtype=np.float64)
    return np.zeros_like(x, dtype=np.float64)


def _eval_components_numba(x, peaks):
    n = len(peaks)
    comps = np.empty((n, x.size), dtype=np.float64)
    for i, (h, c, w, eta) in enumerate(peaks):
        comps[i] = _pvoigt_nb(x, h, c, w, eta)
    return comps


def _eval_total_numba(x, peaks):
    comps = _eval_components_numba(x, peaks)
    if comps.size:
        return np.sum(comps, axis=0, dtype=np.float64)
    return np.zeros_like(x, dtype=np.float64)


def _eval_components_gpu(x, peaks):  # pragma: no cover - GPU path
    xp = _cp
    x_cp = xp.asarray(x, dtype=xp.float64)
    comps = []
    for h, c, w, eta in peaks:
        comps.append(_pvoigt_cp(x_cp, h, c, w, eta))
    if comps:
        C = xp.stack(comps)
        return np.asarray(xp.asnumpy(C), dtype=np.float64)
    return np.zeros((0, x_cp.size), dtype=np.float64)


def _eval_total_gpu(x, peaks):  # pragma: no cover - GPU path
    xp = _cp
    x_cp = xp.asarray(x, dtype=xp.float64)
    y = xp.zeros_like(x_cp, dtype=xp.float64)
    for h, c, w, eta in peaks:
        y += _pvoigt_cp(x_cp, h, c, w, eta)
    return np.asarray(xp.asnumpy(y), dtype=np.float64)


def _eval_components_backend(x, peaks):
    if _BACKEND == "cupy":
        return _eval_components_gpu(x, peaks)
    if _BACKEND == "numba":
        return _eval_components_numba(x, peaks)
    return _eval_components_numpy(x, peaks)


def _eval_total_backend(x, peaks):
    if _BACKEND == "cupy":
        return _eval_total_gpu(x, peaks)
    if _BACKEND == "numba":
        return _eval_total_numba(x, peaks)
    return _eval_total_numpy(x, peaks)


# ---------------------------------------------------------------------------
# Public evaluators
def eval_total(x: np.ndarray, peaks) -> np.ndarray:
    """Evaluate the sum of pseudo-Voigt peaks on ``x``."""

    x = np.asarray(x, dtype=np.float64)
    fast = _eval_total_backend(x, peaks)
    if _SHADOW_COMPARE and _BACKEND != "numpy":
        ref = _eval_total_numpy(x, peaks)
        if not np.allclose(fast, ref, rtol=_SHADOW_RTOL, atol=_SHADOW_ATOL):
            _shadow_fail()
            return ref
    return fast.astype(np.float64, copy=False)


def eval_components(x: np.ndarray, peaks) -> np.ndarray:
    """Return component matrix with shape ``(n_peaks, len(x))``."""

    x = np.asarray(x, dtype=np.float64)
    fast = _eval_components_backend(x, peaks)
    if _SHADOW_COMPARE and _BACKEND != "numpy":
        ref = _eval_components_numpy(x, peaks)
        if not np.allclose(fast, ref, rtol=_SHADOW_RTOL, atol=_SHADOW_ATOL):
            _shadow_fail()
            return ref
    return fast.astype(np.float64, copy=False)


def design_matrix(x: np.ndarray, peaks) -> np.ndarray:
    """Return design matrix ``A`` with columns per peak (shape ``m×n``)."""

    comps = eval_components(x, peaks)
    return comps.T.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Warmups to trigger JIT/initial allocations
def _warmup_numba() -> None:  # pragma: no cover - warmup only
    if not _NUMBA_OK:
        return
    x = np.linspace(0.0, 1.0, 8)
    _ = _pvoigt_nb(x, 1.0, 0.5, 0.1, 0.5)


def _warmup_gpu() -> None:  # pragma: no cover - warmup only
    if not _CUPY_OK:
        return
    x = _cp.linspace(0.0, 1.0, 8)
    _ = _pvoigt_cp(x, 1.0, 0.5, 0.1, 0.5)


__all__ = [
    "set_numba",
    "set_gpu",
    "set_cache_baseline",
    "set_seed",
    "set_max_workers",
    "set_unc_workers",
    "get_max_workers",
    "get_unc_workers",
    "set_gpu_chunk",
    "set_logger",
    "which_backend",
    "eval_total",
    "eval_components",
    "design_matrix",
    "enable_shadow_compare",
    "cache_baseline_enabled",
    "apply_global_seed",
    "blas_single_thread_ctx",
    "blas_limit_ctx",
    "get_parallel_config",
    "note_tpctl_absence_once",
]

