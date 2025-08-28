from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Callable, Dict, Any

import numpy as np

try:
    from .data_io import pseudo_voigt  # type: ignore
except Exception:  # fallback when data_io lacks pseudo_voigt
    def pseudo_voigt(x, height, x0, fwhm, eta):
        x = np.asarray(x, float)
        dx = (x - x0) / float(fwhm)
        A = 4.0 * np.log(2.0)
        gaussian = np.exp(-A * dx * dx)
        lorentz = 1.0 / (1.0 + 4.0 * dx * dx)
        return height * ((1.0 - eta) * gaussian + eta * lorentz)

from .peaks import Peak
from . import models
from .weights import noise_weights
from .residuals import build_residual, jacobian_fd, build_residual_jac
from fit import orchestrator
from fit.bounds import pack_theta_bounds
from infra import performance
from fit import classic, modern, modern_vp, lmfit_backend
from fit.utils import mad_sigma, robust_cost
from scipy.optimize import least_squares, nnls


@dataclass
class StepResult:
    accepted: bool
    cost0: float
    cost1: float
    step_norm: float
    lambda_used: float | None
    backtracks: int
    reason: str


def _make_predict_full_generic(
    x_all: np.ndarray,
    base_all: np.ndarray,
    add_mode: bool,
    model_peaks: Callable[..., np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def predict_full(theta_vec: np.ndarray) -> np.ndarray:
        y_peaks = model_peaks(x_all, *np.asarray(theta_vec, float))
        return base_all + y_peaks if add_mode else y_peaks

    return predict_full


def _vp_design_columns(
    x: np.ndarray,
    c_list: List[float],
    w_list: List[float],
    eta_list: List[float],
) -> np.ndarray:
    cols = [pseudo_voigt(x, 1.0, c, w, eta) for c, w, eta in zip(c_list, w_list, eta_list)]
    return np.column_stack(cols) if cols else np.zeros((len(x), 0), float)


def _vp_unpack_cw(theta_nl: np.ndarray, struct: List[dict], peaks) -> (List[float], List[float]):
    theta_nl = list(np.asarray(theta_nl, float).ravel())
    c_list, w_list = [], []
    tptr = 0
    for s, pk in zip(struct, peaks):
        if s.get("ic") is not None:
            c = float(theta_nl[tptr])
            tptr += 1
        else:
            c = float(s.get("c_fixed", pk.center))
        if s.get("iw") is not None:
            w = float(theta_nl[tptr])
            tptr += 1
        else:
            w = float(s.get("w_fixed", pk.fwhm))
        c_list.append(c)
        w_list.append(w)
    return c_list, w_list


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

    def model_peaks(x_data: np.ndarray, *theta_vec: np.ndarray) -> np.ndarray:
        pk = []
        for i in range(len(peaks_out)):
            c, h, fw, eta = theta_vec[4 * i : 4 * i + 4]
            pk.append((h, c, fw, eta))
        return performance.eval_total(x_data, pk)

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

    names: list[str] = []
    locked: list[bool] = []
    for i, pk in enumerate(peaks_out):
        names.extend([f"center{i+1}", f"height{i+1}", f"fwhm{i+1}", f"eta{i+1}"])
        locked.extend([pk.lock_center, False, pk.lock_width, False])

    if return_jacobian:
        r0 = resid_fn(theta)
        J = jacobian_fd(resid_fn, theta)
        dof = max(r0.size - theta.size, 1)

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

    base_all = baseline if baseline is not None else np.zeros_like(x)
    add_mode = bool(mode == "add")
    solver_used = best_res.solver if best_res is not None else solver_name
    if solver_used in ("modern_vp", "lmfit_vp"):
        solver_kind = "vp"
    elif solver_used == "modern_trf":
        solver_kind = "trf"
    elif solver_used == "classic":
        solver_kind = "classic"
    elif solver_used.startswith("lmfit"):
        solver_kind = "lmfit"
    else:
        solver_kind = str(solver_used)

    fit_ctx: Dict[str, Any] = {
        "x_all": np.asarray(x, float),
        "base_all": np.asarray(base_all, float),
        "add_mode": add_mode,
        "theta_hat": np.asarray(theta, float),
        "param_names": list(names),
        "predict_full": _make_predict_full_generic(
            np.asarray(x, float),
            np.asarray(base_all, float),
            add_mode,
            model_peaks,
        ),
        "solver_kind": solver_kind,
    }
    if solver_kind == "vp":
        struct: List[dict] = []
        idx = 0
        for pk in peaks_out:
            s: dict = {}
            if pk.lock_center:
                s["ic"] = None
                s["c_fixed"] = pk.center
            else:
                s["ic"] = idx
                idx += 1
            if pk.lock_width:
                s["iw"] = None
                s["w_fixed"] = pk.fwhm
            else:
                s["iw"] = idx
                idx += 1
            struct.append(s)
        fit_ctx.update(
            {
                "vp_struct": struct,
                "vp_etas": [p.eta for p in peaks_out],
                "x_fit": np.asarray(x_fit, float),
                "y_target_fit": np.asarray(y_target, float),
                "wvec_fit": None if weights is None else np.asarray(weights, float),
                "mask": np.asarray(mask, bool),
                "vp_unpack_cw": lambda theta_nl, struct=struct, peaks=peaks_out: _vp_unpack_cw(theta_nl, struct, peaks),
            }
        )
    result["fit_ctx"] = fit_ctx

    if return_predictors:
        x_all = x
        baseline_all = baseline
        x_fit = x[mask]
        baseline_fit = baseline[mask] if (baseline is not None and mode == "add") else None

        predict_full = fit_ctx["predict_full"]

        def predict_fit(th: np.ndarray) -> np.ndarray:
            pk = []
            for i in range(len(peaks_out)):
                c, h, w, e = th[4 * i : 4 * (i + 1)]
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


# ---- Step adapters ------------------------------------------------------


def classic_step(payload: dict) -> Tuple[np.ndarray, StepResult]:
    """Perform one Classic solver iteration."""

    x = np.asarray(payload["x"], float)
    y = np.asarray(payload["y"], float)
    peaks = payload.get("peaks", [])
    mode = payload.get("mode", "add")
    baseline = payload.get("baseline")
    opts = dict(payload.get("options", {}))

    init = classic.prepare_state(x, y, peaks, mode, baseline, opts)
    state = init["state"]
    cost0 = float(init.get("cost", 0.0))
    state, accepted, cost0, cost1, info = classic.iterate(state)
    theta = state.get("theta", np.array([]))
    res = StepResult(
        accepted=bool(info.get("accepted", False)),
        cost0=float(cost0),
        cost1=float(cost1 if accepted else cost0),
        step_norm=float(info.get("step_norm", 0.0)),
        lambda_used=float(info.get("lambda", 0.0)),
        backtracks=int(info.get("backtracks", 0)),
        reason=str(info.get("reason", "ok" if accepted else "no_decrease")),
    )
    return theta, res


def _trf_prepare(payload: dict):
    x = np.asarray(payload["x"], float)
    y = np.asarray(payload["y"], float)
    peaks = payload.get("peaks", [])
    mode = payload.get("mode", "add")
    baseline = payload.get("baseline")
    opts = dict(payload.get("options", {}))

    theta_full0, bounds_full = pack_theta_bounds(peaks, x, opts)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(opts.get("min_fwhm", 1e-6)), 2.0 * dx_med)
    theta0, bounds, x_scale, indices = modern._to_solver_vectors(
        theta_full0, bounds_full, peaks, fwhm_min
    )
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weights = noise_weights(y_target, opts.get("weights", "none"))
    resid_jac = build_residual_jac(x, y, peaks, mode, baseline, weights)
    return theta0, bounds, x_scale, indices, theta_full0, resid_jac, opts


def modern_trf_step(payload: dict) -> Tuple[np.ndarray, StepResult]:
    theta0, bounds, x_scale, indices, theta_full0, resid_jac, opts = _trf_prepare(payload)
    loss = opts.get("loss", "linear")
    f_scale = float(opts.get("f_scale", 1.0))
    r0, _ = resid_jac(theta0)
    if loss != "linear" and (not np.isfinite(f_scale) or f_scale <= 0):
        sigma = mad_sigma(r0)
        f_scale = max(sigma, 1e-12)
    cost0 = robust_cost(r0, loss, f_scale)

    def fun(t):
        r, _ = resid_jac(t)
        return r

    def jac(t):
        _, J = resid_jac(t)
        return J

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

    r1, _ = resid_jac(res.x)
    cost1 = robust_cost(r1, loss, f_scale)
    delta = res.x - theta0
    step_norm = float(np.linalg.norm(delta))
    accepted = np.isfinite(cost1) and cost1 < cost0 and step_norm > 1e-14

    theta_full = theta_full0.copy()
    if accepted:
        theta_full[indices] = res.x
    else:
        cost1 = cost0
        step_norm = 0.0

    result = StepResult(
        accepted=accepted,
        cost0=float(cost0),
        cost1=float(cost1),
        step_norm=step_norm,
        lambda_used=None,
        backtracks=0,
        reason="ok" if accepted else "no_decrease",
    )
    return theta_full, result


def _vp_initial_cost(x, y, peaks, baseline, opts):
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weights = noise_weights(y_target, opts.get("weights", "none"))
    unit = [(1.0, p.center, p.fwhm, p.eta) for p in peaks]
    A = performance.design_matrix(x, unit)
    if weights is not None:
        Aw = A * weights[:, None]
        bw = y_target * weights
    else:
        Aw = A
        bw = y_target
    h, _ = nnls(Aw, bw)
    h = np.minimum(h, opts.get("max_height", np.inf))
    r = A @ h - y_target
    if weights is not None:
        r = r * weights
    loss = opts.get("loss", "linear")
    f_scale_opt = float(opts.get("f_scale", 0.0))
    sigma = 1.0
    if loss != "linear" and f_scale_opt <= 0:
        sigma = mad_sigma(r)
        fs = max(sigma, 1e-12)
    else:
        fs = f_scale_opt if f_scale_opt > 0 else sigma
    cost = robust_cost(r, loss, fs)
    return cost


def modern_vp_step(payload: dict) -> Tuple[np.ndarray, StepResult]:
    x = np.asarray(payload["x"], float)
    y = np.asarray(payload["y"], float)
    peaks = payload.get("peaks", [])
    mode = payload.get("mode", "add")
    baseline = payload.get("baseline")
    opts = dict(payload.get("options", {}))

    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weight_mode = opts.get("weights", "none")
    weights = noise_weights(y_target, weight_mode)
    p95 = float(np.percentile(np.abs(y_target), 95)) if y_target.size else 1.0
    max_height_factor = float(opts.get("max_height_factor", np.inf))
    opts["max_height"] = max_height_factor * p95
    opts["max_fwhm"] = opts.get("max_fwhm", 0.5 * (x.max() - x.min()))

    cost0 = _vp_initial_cost(x, y, peaks, baseline, opts)
    theta0_full, _ = pack_theta_bounds(peaks, x, opts)

    opts1 = dict(opts)
    opts1["maxfev"] = 1
    res = modern_vp.solve(x, y, [Peak(p.center, p.height, p.fwhm, p.eta) for p in peaks], mode, baseline, opts1)
    theta_full = res.get("theta", theta0_full)
    cost1 = float(res.get("cost", cost0))
    delta = theta_full - theta0_full
    step_norm = float(np.linalg.norm(delta))
    accepted = np.isfinite(cost1) and cost1 < cost0 and step_norm > 1e-14
    if not accepted:
        theta_full = theta0_full
        cost1 = cost0
        step_norm = 0.0
    result = StepResult(
        accepted=accepted,
        cost0=float(cost0),
        cost1=float(cost1),
        step_norm=step_norm,
        lambda_used=None,
        backtracks=int(bool(res.get("meta", {}).get("backtracked", False))),
        reason="ok" if accepted else "no_decrease",
    )
    return theta_full, result


def lmfit_step(payload: dict) -> Tuple[np.ndarray, StepResult]:
    x = np.asarray(payload["x"], float)
    y = np.asarray(payload["y"], float)
    peaks = payload.get("peaks", [])
    mode = payload.get("mode", "add")
    baseline = payload.get("baseline")
    opts = dict(payload.get("options", {}))

    base = baseline if baseline is not None else 0.0
    y_target = y - base
    weight_mode = opts.get("weights", "none")
    weights = noise_weights(y_target, weight_mode)
    p95 = float(np.percentile(np.abs(y_target), 95)) if y_target.size else 1.0
    max_height_factor = float(opts.get("max_height_factor", np.inf))
    opts["max_height"] = max_height_factor * p95
    opts["max_fwhm"] = opts.get("max_fwhm", 0.5 * (x.max() - x.min()))

    cost0 = _vp_initial_cost(x, y, peaks, baseline, opts)
    theta0_full, _ = pack_theta_bounds(peaks, x, opts)

    opts1 = dict(opts)
    opts1["maxfev"] = 1
    res = lmfit_backend.solve(
        x,
        y,
        [Peak(p.center, p.height, p.fwhm, p.eta) for p in peaks],
        mode,
        baseline,
        opts1,
    )
    theta_full = res.get("theta", theta0_full)
    cost1 = float(res.get("cost", cost0))
    delta = theta_full - theta0_full
    step_norm = float(np.linalg.norm(delta))
    accepted = np.isfinite(cost1) and cost1 < cost0 and step_norm > 1e-14
    if not accepted:
        theta_full = theta0_full
        cost1 = cost0
        step_norm = 0.0
    result = StepResult(
        accepted=accepted,
        cost0=float(cost0),
        cost1=float(cost1),
        step_norm=step_norm,
        lambda_used=None,
        backtracks=0,
        reason="ok" if accepted else "no_decrease",
    )
    return theta_full, result


