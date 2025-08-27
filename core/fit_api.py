from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from .peaks import Peak
from . import models
from .weights import noise_weights
from .residuals import build_residual, jacobian_fd
from fit import orchestrator
from fit.bounds import pack_theta_bounds
from infra import performance


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
    return_jacobian: bool = False,
    return_predictors: bool = False,
) -> dict:
    """Run a fit mirroring the single-file GUI path.

    Parameters
    ----------
    return_jacobian:
        When ``True`` the return dictionary is extended with details useful
        for uncertainty estimation.  The keys ``"jacobian"`` and
        ``"residual"`` are added together with parameter meta data.
    return_predictors:
        When also ``True`` prediction helpers ``predict_fit`` and
        ``predict_full`` are included.  These mirror the behaviour of the
        model used during fitting and are suitable for uncertainty band
        calculations.
    """

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

    def ymodel_fn(th: np.ndarray) -> np.ndarray:
        pk = []
        for i in range(len(peaks_out)):
            c, h, fw, eta = th[4 * i : 4 * i + 4]
            pk.append((h, c, fw, eta))
        total = performance.eval_total(x, pk)
        if baseline is not None and mode == "add":
            total = total + baseline
        return total

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

    result = {
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

    if return_jacobian:
        r0 = resid_fn(theta)
        J = jacobian_fd(resid_fn, theta)
        dof = max(r0.size - theta.size, 1)

        # parameter meta information
        names: list[str] = []
        locked: list[bool] = []
        for i, pk in enumerate(peaks_out):
            names.extend([f"center{i+1}", f"height{i+1}", f"fwhm{i+1}", f"eta{i+1}"])
            locked.extend([pk.lock_center, False, pk.lock_width, False])

        result.update(
            {
                "jacobian": J,
                "residual": r0,
                "param_names": names,
                "locked_mask": np.array(locked, bool),
                "bounds": (lo, hi),
                "mask_fit": mask,
                "rss": float(np.dot(r0, r0)),
                "dof": dof,
                "x": x,
                "baseline": baseline,
                "mode": mode,
                "x_all": x,
                "base_all": baseline,
                "add_mode": mode == "add",
                "residual_fn": resid_fn,
                "ymodel_fn": ymodel_fn,
            }
        )

        if return_predictors:
            x_all = x
            baseline_all = baseline
            x_fit = x[mask]
            baseline_fit = baseline[mask] if (baseline is not None and mode == "add") else None

            def predict_full(th: np.ndarray) -> np.ndarray:
                return ymodel_fn(th)

            def predict_fit(th: np.ndarray) -> np.ndarray:
                pk = []
                for i in range(len(peaks_out)):
                    c, h, w, e = th[4 * i : 4 * i + 4]
                    pk.append((h, c, w, e))
                total = performance.eval_total(x_fit, pk)
                if baseline_fit is not None:
                    total = total + baseline_fit
                return total

            result.update({"predict_fit": predict_fit, "predict_full": predict_full})

    return result


def _loss_weights(r: np.ndarray, loss: str, f_scale: float) -> np.ndarray:
    """Return square-root robust loss weights for residual ``r``.

    Parameters
    ----------
    r:
        Weighted residual vector.
    loss:
        Loss name (``linear``, ``soft_l1``, ``huber`` or ``cauchy``).
    f_scale:
        Robust loss scale parameter.
    """

    if loss == "linear" or f_scale <= 0:
        return np.ones_like(r)
    z = r / float(f_scale)
    if loss == "soft_l1":
        return (1.0 / np.sqrt(1.0 + z * z)) ** 0.5
    if loss == "huber":
        w = np.ones_like(z)
        mask = np.abs(z) > 1.0
        w[mask] = 1.0 / np.sqrt(np.abs(z[mask]))
        return w
    if loss == "cauchy":
        return 1.0 / np.sqrt(1.0 + z * z)
    return np.ones_like(r)


def build_residual_and_jacobian(payload: dict, solver_choice: str) -> dict:
    """Construct residual/Jacobian consistent with the solver settings.

    Parameters
    ----------
    payload:
        Dictionary containing ``x``, ``y``, ``peaks``, ``mode``, ``baseline`` and
        ``options``.  ``options`` may include ``loss``, ``weights`` and
        ``f_scale``.
    solver_choice:
        Name of the active solver (``classic``, ``modern_trf`` or ``lmfit``).

    Returns
    -------
    dict
        ``theta`` initial parameters, ``bounds`` tuple, ``locked_mask`` boolean
        mask and ``residual_jac`` callable returning weighted residuals and
        Jacobian for a given ``theta``.
    """

    x = np.asarray(payload["x"], float)
    y = np.asarray(payload["y"], float)
    peaks = payload.get("peaks", [])
    baseline = payload.get("baseline")
    mode = payload.get("mode", "add")
    opts = dict(payload.get("options", {}))

    theta0, (lo, hi) = pack_theta_bounds(peaks, x, opts)

    # locked mask in GUI order [c, h, w, e]*n
    locked: list[bool] = []
    for p in peaks:
        locked.extend([p.lock_center, False, p.lock_width, False])
    locked_mask = np.array(locked, dtype=bool)

    # base residual without any weighting / robust scaling
    resid_base = build_residual(x, y, peaks, mode, baseline, "linear", None)

    weights = noise_weights(y - (baseline if baseline is not None and mode == "add" else 0.0), opts.get("weights", "none"))
    loss = opts.get("loss", "linear")
    f_scale = float(opts.get("f_scale", 1.0))

    def residual_jac(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = resid_base(theta)
        J = jacobian_fd(resid_base, theta)
        if weights is not None:
            r = r * weights
            J = J * weights[:, None]
        lw = _loss_weights(r, loss, f_scale)
        r = r * lw
        J = J * lw[:, None]
        return r, J

    return {
        "theta": theta0,
        "bounds": (lo, hi),
        "locked_mask": locked_mask,
        "residual_jac": residual_jac,
    }


@dataclass
class StepResult:
    accepted: bool
    cost0: float
    cost1: float
    step_norm: float
    lambda_used: float | None
    backtracks: int
    reason: str


def _map_reason(reason: str) -> str:
    if reason in ("accepted", "ok"):
        return "ok"
    if reason == "nonfinite":
        return "nan_guard"
    return reason


def step_classic(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
):
    from fit import classic

    prep = classic.prepare_state(x, y, peaks, mode, baseline, options)
    state = prep["state"]
    cost0 = prep["cost"]
    state, ok, c0, c1, info = classic.iterate(state)
    theta = state["theta"]
    peaks_out = state["peaks"]
    res = StepResult(
        accepted=ok,
        cost0=float(c0),
        cost1=float(c1),
        step_norm=float(info.get("step_norm", 0.0)),
        lambda_used=float(info.get("lambda", 0.0)),
        backtracks=int(info.get("backtracks", 0)),
        reason=_map_reason(info.get("reason", "ok")),
    )
    return theta, peaks_out, res


def step_modern_vp(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
):
    from fit import modern_vp

    prep = modern_vp.prepare_state(x, y, peaks, mode, baseline, options)
    state = prep["state"]
    state, ok, c0, c1, info = modern_vp.iterate(state)
    theta = state["theta"]
    peaks_out = state["peaks"]
    res = StepResult(
        accepted=ok,
        cost0=float(c0),
        cost1=float(c1),
        step_norm=float(info.get("step_norm", 0.0)),
        lambda_used=float(info.get("lambda", 0.0)),
        backtracks=int(info.get("backtracks", 0)),
        reason=_map_reason(info.get("reason", "ok")),
    )
    return theta, peaks_out, res


def step_modern_trf(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
):
    from scipy.optimize import least_squares
    from fit.modern import _to_solver_vectors
    from fit.utils import robust_cost

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    f_scale = float(options.get("f_scale", 1.0))

    base_arr = base if base is not None else 0.0
    y_target = y - base_arr
    weights = noise_weights(y_target, weight_mode)

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)
    theta0, bounds, x_scale, indices = _to_solver_vectors(
        theta0_full, bounds_full, peaks, fwhm_min
    )

    from core.residuals import build_residual_jac as _brj
    resid_jac_fn = _brj(x, y, peaks, mode, base, weights)

    def fun(t):
        r, _ = resid_jac_fn(t)
        return r

    def jac(t):
        _, J = resid_jac_fn(t)
        return J

    r0, _ = resid_jac_fn(theta0)
    cost0 = robust_cost(r0, loss, f_scale)
    res = least_squares(
        fun,
        theta0,
        jac=jac,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        bounds=bounds,
        x_scale=x_scale,
        max_nfev=2,
    )
    r1 = res.fun
    cost1 = robust_cost(r1, loss, f_scale)
    theta_full = theta0_full.copy()
    theta_full[indices] = res.x
    step_norm = float(np.linalg.norm(res.x - theta0))
    accepted = np.isfinite(cost1) and cost1 < cost0 and step_norm > 1e-14
    reason = "ok" if accepted else "no_decrease"
    if not accepted:
        theta_full = theta0_full
        cost1 = cost0
        step_norm = 0.0

    peaks_out: List[Peak] = []
    for i, pk in enumerate(peaks):
        c, h, w, e = theta_full[4 * i : 4 * (i + 1)]
        peaks_out.append(Peak(c, h, w, e, pk.lock_center, pk.lock_width))

    diag = StepResult(
        accepted=accepted,
        cost0=float(cost0),
        cost1=float(cost1),
        step_norm=step_norm,
        lambda_used=None,
        backtracks=0,
        reason=_map_reason(reason),
    )
    return theta_full, peaks_out, diag


def step_lmfit_vp(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
):
    try:
        from fit import lmfit_backend
    except Exception as exc:  # pragma: no cover - optional dependency missing
        theta0, _ = pack_theta_bounds(peaks, x, options)
        diag = StepResult(
            accepted=False,
            cost0=float("nan"),
            cost1=float("nan"),
            step_norm=0.0,
            lambda_used=None,
            backtracks=0,
            reason=str(exc),
        )
        return theta0, peaks, diag

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(baseline, float) if baseline is not None else None

    res = lmfit_backend.solve(x, y, peaks, mode, base, {**options, "maxfev": 1})
    theta1 = res["theta"]
    peaks_out = [
        Peak(theta1[4 * i], theta1[4 * i + 1], theta1[4 * i + 2], theta1[4 * i + 3])
        for i in range(len(peaks))
    ]

    # Estimate cost0 via residual before step
    resid0 = build_residual(x, y, peaks, mode, base, "linear", None)
    theta0, _ = pack_theta_bounds(peaks, x, options)
    r0 = resid0(theta0)
    cost0 = 0.5 * float(r0 @ r0)
    cost1 = res.get("cost", cost0)
    step_norm = float(np.linalg.norm(theta1 - theta0))
    accepted = np.isfinite(cost1) and cost1 < cost0 and step_norm > 1e-14
    if not accepted:
        theta1 = theta0
        peaks_out = peaks
        cost1 = cost0
        step_norm = 0.0

    diag = StepResult(
        accepted=accepted,
        cost0=float(cost0),
        cost1=float(cost1),
        step_norm=step_norm,
        lambda_used=None,
        backtracks=0,
        reason="ok" if accepted else "no_decrease",
    )
    return theta1, peaks_out, diag


STEP_DISPATCH = {
    "classic": step_classic,
    "modern_trf": step_modern_trf,
    "modern_vp": step_modern_vp,
    "lmfit_vp": step_lmfit_vp,
}


def step_router(
    solver: str,
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
):
    func = STEP_DISPATCH.get(solver)
    if func is None:
        raise ValueError(f"unknown solver '{solver}'")
    return func(x, y, peaks, mode, baseline, options)

