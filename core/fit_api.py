from __future__ import annotations

import copy
from dataclasses import replace
from typing import List, Optional

import numpy as np

from .peaks import Peak
from . import models
from .weights import noise_weights
from .residuals import build_residual
from fit import orchestrator
from fit.bounds import pack_theta_bounds


def run_fit_consistent(
    x: np.ndarray,
    y: np.ndarray,
    peaks_in: List[Peak],
    cfg: dict,
    baseline: Optional[np.ndarray],
    mode: str,
    fit_mask: np.ndarray,
    reheight: bool = False,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run a fit mirroring the single-file GUI path."""

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(fit_mask, bool)
    if mask.shape != x.shape:
        raise ValueError("fit_mask shape mismatch")
    if baseline is not None:
        baseline = np.asarray(baseline, float)
        if baseline.shape != x.shape:
            raise ValueError("baseline shape mismatch")

    peaks0 = copy.deepcopy(peaks_in)
    if reheight:
        sig = y - (baseline if baseline is not None else 0.0)
        med = float(np.median(sig))
        for i, pk in enumerate(peaks0):
            if getattr(pk, "lock_height", False):
                continue
            y_at = float(np.interp(pk.center, x, sig))
            peaks0[i] = replace(pk, height=max(y_at - med, 1e-6))

    x_fit = x[mask]
    y_fit = y[mask]
    base_fit = baseline[mask] if baseline is not None else None

    solver_name = cfg.get("solver", "modern_vp")
    solver_opts = dict(cfg.get(solver_name, {}))
    opts = dict(solver_opts)
    opts.update(
        {
            "solver": solver_name,
            "loss": cfg.get("solver_loss", opts.get("loss", "linear")),
            "weights": cfg.get("solver_weight", opts.get("weights", "none")),
            "f_scale": cfg.get("solver_fscale", opts.get("f_scale", 1.0)),
            "maxfev": cfg.get("solver_maxfev", opts.get("maxfev", 20000)),
            "restarts": 1,
            "jitter_pct": 0.0,
        }
    )

    p0, (lo, hi) = pack_theta_bounds(peaks0, x_fit, opts)

    restarts = int(cfg.get("solver_restarts", 1))
    jitter_pct = float(cfg.get("solver_jitter_pct", 0.0))
    perf_seed = bool(cfg.get("perf_seed_all", False))
    base_seed = rng_seed if perf_seed else None

    y_target = y_fit - (base_fit if base_fit is not None else 0.0)
    weights = noise_weights(y_target, opts.get("weights", "none"))

    best_res = None
    best_theta = None
    best_rmse = np.inf
    for r in range(max(1, restarts)):
        seed = (base_seed + r) if base_seed is not None else None
        rng = np.random.default_rng(seed)
        theta_start = p0.copy()
        if jitter_pct:
            for i, pk in enumerate(peaks0):
                if not pk.lock_center:
                    theta_start[4 * i] += (
                        theta_start[4 * i + 2]
                        * (jitter_pct / 100.0)
                        * rng.standard_normal()
                    )
                theta_start[4 * i + 1] *= 1.0 + (jitter_pct / 100.0) * rng.standard_normal()
                if not pk.lock_width:
                    theta_start[4 * i + 2] *= 1.0 + (
                        jitter_pct / 100.0
                    ) * rng.standard_normal()
            theta_start = np.clip(theta_start, lo, hi)

        peaks_start = []
        for i, pk in enumerate(peaks0):
            c, h, w, e = theta_start[4 * i : 4 * (i + 1)]
            peaks_start.append(Peak(c, h, w, e, pk.lock_center, pk.lock_width))

        y_solver = y_target if mode == "subtract" else y_fit
        base_solver = None if mode == "subtract" else base_fit
        res = orchestrator.run_fit_with_fallbacks(
            x_fit, y_solver, peaks_start, mode, base_solver, opts
        )
        theta = np.asarray(res.theta, float)
        resid_fn = build_residual(x_fit, y_fit, res.peaks_out, mode, base_fit, "linear", None)
        rmse = float(np.sqrt(np.mean(resid_fn(theta) ** 2))) if theta.size else float("nan")
        if rmse < best_rmse:
            best_rmse = rmse
            best_theta = theta.copy()
            best_res = res

    theta = best_theta if best_theta is not None else p0.copy()
    clipped = np.clip(theta, lo, hi)
    clipped_after = bool(np.any(np.abs(clipped - theta) > 1e-12))
    theta = clipped
    for i, pk in enumerate(peaks0):
        if pk.lock_center:
            theta[4 * i] = p0[4 * i]
        if pk.lock_width:
            theta[4 * i + 2] = p0[4 * i + 2]

    assert np.all(theta >= lo - 1e-12) and np.all(theta <= hi + 1e-12)

    peaks_out: List[Peak] = []
    for i, pk in enumerate(peaks0):
        c, h, w, e = theta[4 * i : 4 * (i + 1)]
        peaks_out.append(
            Peak(c, h, w, e, pk.lock_center, pk.lock_width)
        )

    resid_fn = build_residual(x_fit, y_fit, peaks_out, mode, base_fit, "linear", None)
    rmse = float(np.sqrt(np.mean(resid_fn(theta) ** 2))) if theta.size else float("nan")

    baseline_cfg = cfg.get("baseline", {})
    baseline_params = {
        "lam": baseline_cfg.get("lam"),
        "p": baseline_cfg.get("p"),
        "niter": baseline_cfg.get("niter"),
        "thresh": baseline_cfg.get("thresh"),
        "thresh_hit": bool(baseline_cfg.get("thresh_hit", False)),
    }

    if verbose:
        xmin = float(x_fit[0]) if x_fit.size else float("nan")
        xmax = float(x_fit[-1]) if x_fit.size else float("nan")
        print(
            "mask_len=%d window=[%g,%g] bounds=(%g,%g) p0_head=%s p0_tail=%s restarts=%d jitter_pct=%g rmse=%g clipped=%s"
            % (
                mask.sum(),
                xmin,
                xmax,
                lo.min() if lo.size else np.nan,
                hi.max() if hi.size else np.nan,
                p0[:3],
                p0[-3:] if p0.size >= 3 else p0,
                restarts,
                jitter_pct,
                rmse,
                clipped_after,
            )
        )

    return {
        "peaks_out": peaks_out,
        "rmse": rmse,
        "theta": theta,
        "bounds": (lo, hi),
        "p0": p0,
        "mask_len": int(mask.sum()),
        "baseline_params": baseline_params,
        "restarts_used": restarts,
        "jitter_pct": jitter_pct,
        "clipped_after_solve": clipped_after,
        "fit_ok": bool(best_res.success if best_res is not None else False),
    }

