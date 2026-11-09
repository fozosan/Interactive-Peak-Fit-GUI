from __future__ import annotations

import copy
import inspect
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
from . import noise_model
from . import baseline_model
from .residuals import build_residual, jacobian_fd, build_residual_jac
from . import signals
from .signals import polynomial_baseline
from .param_utils import vp_unpack_centers_widths
from fit import orchestrator, solve as fit_solve
from fit.bounds import height_cap, held_heights, pack_theta_bounds
from infra import performance
from fit import classic, modern, modern_vp, lmfit_backend
from fit.utils import mad_sigma, robust_cost
from scipy.optimize import least_squares


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


def _params_on_bounds(theta, lo, hi, locked, names) -> List[str]:
    """Names of the free parameters that sit on a bound of the fit."""

    theta = np.asarray(theta, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    on_lo = theta <= lo + 1e-6 * np.maximum(1.0, np.abs(lo))
    on_hi = theta >= hi - 1e-6 * np.maximum(1.0, np.abs(hi))
    return [n for n, hit, fixed in zip(names, on_lo | on_hi, locked) if hit and not fixed]


def _vp_design_columns(
    x: np.ndarray,
    c_list: List[float],
    w_list: List[float],
    eta_list: List[float],
) -> np.ndarray:
    cols = [pseudo_voigt(x, 1.0, c, w, eta) for c, w, eta in zip(c_list, w_list, eta_list)]
    return np.column_stack(cols) if cols else np.zeros((len(x), 0), float)


# _vp_unpack_cw is now handled by vp_unpack_centers_widths in param_utils

# cfg["noise_model"]: "none" fits every point with the same weight (the core
# default); "per_spectrum" estimates var_i = s0^2 + g * S_i from a first fit's
# residuals and refits with weights 1 / sigma_i (game plan 1.5, decision D1).
NOISE_MODELS = ("none", "per_spectrum")


def _noise_model_choice(cfg: dict) -> str:
    raw = (cfg or {}).get("noise_model", "none")
    if raw is None or raw is False:
        return "none"
    if raw is True:
        return "per_spectrum"
    choice = str(raw).strip().lower()
    if choice in ("", "none", "off"):
        return "none"
    if choice not in NOISE_MODELS:
        raise ValueError(f"Unknown noise_model {raw!r}; expected one of {NOISE_MODELS}")
    return choice


def _fixed_noise_sigma(cfg: dict, x: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """``cfg["noise_sigma"]`` on the fit window: one sigma per point of ``x`` or
    of the window, positive and finite. None when not given."""

    raw = (cfg or {}).get("noise_sigma")
    if raw is None:
        return None
    sigma = np.asarray(raw, float).reshape(-1)
    if sigma.size == x.size:
        sigma = sigma[mask]
    elif sigma.size != int(np.count_nonzero(mask)):
        raise ValueError(
            f"noise_sigma has {sigma.size} values for {x.size} points ({int(np.count_nonzero(mask))} in the window)"
        )
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("noise_sigma must be positive and finite")
    return sigma



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

    Noise model (game plan 1.5, decision D1), from ``cfg``:

    - ``cfg["noise_model"]``: ``"none"`` (the default) fits every point with
      the same weight; ``"per_spectrum"`` fits once, estimates var_i = s0^2 +
      g * S_i from that fit's residuals (``core.noise_model``, S_i its signal
      above the baseline) and refits once from there with weights 1 / sigma_i.
    - ``cfg["noise_sigma"]``: fixed sigma_i, one per point of ``x`` or of the
      window; the fit is weighted by them and nothing is estimated (the
      bootstrap refits pass the fit's own).

    The result's ``noise_model`` holds the estimate's provenance (s0, g, n,
    method) or None, and ``noise_sigma`` the sigma_i on the fit window that
    weighted the fit, or None. Residual, Jacobian and rmse stay unweighted;
    the uncertainty methods take ``noise_sigma`` to weight them.

    Baseline (game plan 1.6, decision D2), from ``cfg``:

    - ``cfg["baseline_fit"]``: ``"fixed"`` (the default, the interactive path)
      holds ``baseline`` (or the one ``cfg["baseline"]`` describes) fixed;
      ``"joint"`` fits the baseline with the peaks, as Legendre columns of order
      ``cfg["baseline_order"]`` (default 3) over the fit window
      (``core.baseline_model``), free linear parameters beside the heights. The
      given baseline then only sets the start heights (``reheight``), the fit
      is in add mode, and the noise model's leverage includes the columns.

    The result's ``baseline_fit`` and ``baseline_order`` record the choice,
    ``baseline_coef`` and ``joint_baseline`` the fitted coefficients and their
    columns (None when fixed), and ``baseline`` the baseline the fit used, over
    all of x (the joint one stays at its edge values outside the window).

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

    # Guard: fit_mask must match x shape to avoid accidental range expansion
    if fit_mask is not None:
        mask_arr = np.asarray(fit_mask, bool)
        if mask_arr.shape != x.shape:
            raise ValueError(
                f"fit_mask shape {mask_arr.shape} must match x shape {x.shape}"
            )
        fit_mask = mask_arr

    mask = np.asarray(fit_mask, bool)
    if mask.shape != x.shape:
        raise ValueError("fit_mask shape mismatch")
    noise_choice = _noise_model_choice(cfg)
    joint = baseline_model.baseline_fit_choice(cfg) == "joint"
    joint_order = baseline_model.baseline_order(cfg) if joint else None
    n_baseline = joint_order + 1 if joint else 0

    # Refuse fits the data cannot support before any solver runs. With no peaks,
    # Modern VP hands scipy.optimize.nnls a matrix with no columns and the
    # interpreter aborts. With no more points than free parameters, a solver
    # still reports success, though no degree of freedom is left to judge the
    # fit. No solver fits eta, so each peak has a height plus its centre and
    # width unless they are locked.
    seeds = list(peaks_in or [])
    n_points = int(np.count_nonzero(mask))
    n_free = sum(1 + (not pk.lock_center) + (not pk.lock_width) for pk in seeds) + n_baseline
    skip_reason = None
    if not seeds:
        skip_reason, message = "no_peaks", "no peaks to fit"
    elif n_points <= n_free:
        skip_reason = "too_few_points"
        message = (
            f"{n_points} points in the fit window for {n_free} free parameters"
            if n_points
            else "no points in the fit window"
        )
    if skip_reason is not None:
        return {
            "fit_ok": False,
            "skip_reason": skip_reason,
            "message": message,
            "peaks_out": [
                Peak(pk.center, pk.height, pk.fwhm, pk.eta, pk.lock_center, pk.lock_width)
                for pk in seeds
            ],
            "theta": np.array(
                [v for pk in seeds for v in (pk.center, pk.height, pk.fwhm, pk.eta)], float
            ),
            "rmse": float("nan"),
            "mask_len": n_points,
            "x": x,
            "baseline": baseline,
            "mode": mode,
            "noise_model": None,
            "noise_sigma": None,
            "baseline_fit": "joint" if joint else "fixed",
            "baseline_order": joint_order,
            "baseline_coef": None,
            "joint_baseline": None,
        }

    base_cfg = (cfg.get("baseline") or {})
    method = str(base_cfg.get("method", "als")).strip().lower()
    uses_fit_range = bool(cfg.get("baseline_uses_fit_range", True))
    fit_mask_eff = mask if uses_fit_range else None

    if baseline is not None:
        baseline = np.asarray(baseline, float)
        if baseline.shape != x.shape:
            raise ValueError("baseline shape mismatch")
    elif base_cfg:
        if method == "als":
            lam = float(base_cfg.get("lam", 1e5))
            p = float(base_cfg.get("p", 0.001))
            niter = int(base_cfg.get("niter", 10))
            thresh = float(base_cfg.get("thresh", 0.0))
            if fit_mask_eff is not None and np.any(fit_mask_eff) and not np.all(fit_mask_eff):
                x_sub = x[fit_mask_eff]
                y_sub = y[fit_mask_eff]
                z_sub = signals.als_baseline(
                    y_sub, lam=lam, p=p, niter=niter, tol=thresh
                )
                baseline = np.interp(x, x_sub, z_sub, left=z_sub[0], right=z_sub[-1])
            else:
                baseline = signals.als_baseline(
                    y, lam=lam, p=p, niter=niter, tol=thresh
                )
        elif method == "polynomial":
            deg = int(base_cfg.get("degree", 2))
            norm_x = bool(base_cfg.get("normalize_x", True))
            baseline = polynomial_baseline(
                x, y, degree=deg, mask=fit_mask_eff, normalize_x=norm_x
            )
        else:
            raise ValueError(f"Unknown baseline method: {method}")

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
    jb = None
    B_fit = None
    provisional_fit = None
    if joint:
        # The baseline is fitted with the peaks: the given one has set the start
        # heights and gives the legacy weights their signal; it is not held.
        jb = baseline_model.JointBaseline.over(x_fit, joint_order)
        B_fit = jb.basis(x_fit)
        provisional_fit = baseline[mask] if baseline is not None else None
        baseline = None
        mode = "add"
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
    # What the solver was told, for the exported tables; the weights are named
    # before a noise model or fixed sigma replaces them with numbers.
    solver_used_opts = {
        "loss": opts["loss"],
        "weights": opts["weights"] if isinstance(opts["weights"], str) else "custom",
        "f_scale": opts["f_scale"],
        "maxfev": opts["maxfev"],
    }

    # One height bound for every solver, from the data the peaks are fitted to
    # (fit.bounds.height_cap; with a joint baseline, the data above the
    # computed one): the solver enforces it, and it is reported.
    peak_signal = base_fit if base_fit is not None else provisional_fit
    opts["max_height"] = height_cap(y_fit - (peak_signal if peak_signal is not None else 0.0), opts)
    p0, (lo, hi) = pack_theta_bounds(peaks0, x_fit, opts)
    if joint:
        opts["baseline_basis"] = B_fit
        if isinstance(opts.get("weights"), str) and opts["weights"] != "none":
            # "poisson" / "inv_y" weigh the signal above the baseline: the start one
            w_legacy = noise_weights(
                y_fit - (provisional_fit if provisional_fit is not None else float(np.median(y_fit))),
                opts["weights"],
            )
            if w_legacy is not None:
                opts["weights"] = w_legacy

    restarts = int(cfg.get("solver_restarts", 1))
    jitter_pct = float(cfg.get("solver_jitter_pct", 0.0))
    perf_seed = bool(cfg.get("perf_seed_all", False))
    base_seed = rng_seed if perf_seed else None

    y_target = y_fit - (base_fit if base_fit is not None else 0.0)
    # Fixed per-point sigma (the bootstrap refits pass the fit's) weights the
    # fit as given; otherwise the configured weights, which a per-spectrum
    # noise model replaces below.
    sigma_fit = _fixed_noise_sigma(cfg, x, mask)
    if sigma_fit is not None:
        opts["weights"] = 1.0 / sigma_fit
    weights = noise_weights(y_target, opts.get("weights", "none"))

    y_solver = y_target if mode == "subtract" else y_fit
    base_solver = None if mode == "subtract" else base_fit
    # respect strict disabling of solver fallbacks when requested
    _no_fb = bool((cfg or {}).get("no_solver_fallbacks", False))
    cfg_config = None
    if isinstance(cfg, dict):
        cfg_config = cfg.get("config")
    if not _no_fb and isinstance(cfg_config, dict):
        try:
            _no_fb = bool(cfg_config.get("no_solver_fallbacks", False))
        except Exception:
            _no_fb = _no_fb

    def _run_solver(peaks_start, run_opts):
        if _no_fb:
            res = None
            # Prefer orchestrator.step_once when available and signature-compatible.
            step_once = getattr(orchestrator, "step_once", None)
            if callable(step_once):
                try:
                    sig = inspect.signature(step_once)
                except Exception:
                    sig = None
                expects_state = bool(
                    sig
                    and len(sig.parameters) == 1
                    and all(
                        p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                        for p in sig.parameters.values()
                    )
                )
                if not expects_state:
                    try:
                        res = step_once(x_fit, y_solver, peaks_start, mode, base_solver, run_opts)
                    except TypeError:
                        res = None
            if res is None:
                res = fit_solve(x_fit, y_solver, peaks_start, mode, base_solver, run_opts)
            return res
        return orchestrator.run_fit_with_fallbacks(
            x_fit, y_solver, peaks_start, mode, base_solver, run_opts
        )

    def _fit_baseline(res):
        """The baseline a solve fitted on the window: the given one, or the joint
        columns times that solve's coefficients."""

        if not joint:
            return base_fit
        coef = getattr(res, "baseline_coef", None)
        if coef is None:
            raise RuntimeError(f"the {getattr(res, 'solver', solver_name)} solver returned no baseline coefficients")
        return B_fit @ np.asarray(coef, float)

    def _solve(theta_seed, run_opts, n_restarts, jitter):
        """Best of ``n_restarts`` solves started at ``theta_seed`` (each start
        jittered by ``jitter`` percent), judged by the fit's own weighted
        residuals."""

        w_run = noise_weights(y_target, run_opts.get("weights", "none"))
        best_res = None
        best_theta = None
        best_cost = np.inf
        for r in range(max(1, n_restarts)):
            seed = (base_seed + r) if base_seed is not None else None
            rng = np.random.default_rng(seed)
            theta_start = np.asarray(theta_seed, float).copy()
            if jitter:
                for i, pk in enumerate(peaks0):
                    if not pk.lock_center:
                        theta_start[4 * i] += (
                            theta_start[4 * i + 2]
                            * (jitter / 100.0)
                            * rng.standard_normal()
                        )
                    theta_start[4 * i + 1] *= 1.0 + (jitter / 100.0) * rng.standard_normal()
                    if not pk.lock_width:
                        theta_start[4 * i + 2] *= 1.0 + (
                            jitter / 100.0
                        ) * rng.standard_normal()
                theta_start = np.clip(theta_start, lo, hi)

            peaks_start = []
            for i, pk in enumerate(peaks0):
                c, h, w, e = theta_start[4 * i : 4 * (i + 1)]
                peaks_start.append(Peak(c, h, w, e, pk.lock_center, pk.lock_width))

            res = _run_solver(peaks_start, run_opts)
            theta = np.asarray(res.theta, float)
            resid_fn = build_residual(x_fit, y_fit, res.peaks_out, mode, _fit_baseline(res), "linear", w_run)
            cost = float(np.mean(resid_fn(theta) ** 2)) if theta.size else float("nan")
            if cost < best_cost:
                best_cost = cost
                best_theta = theta.copy()
                best_res = res
        return best_res, best_theta

    def _finalize(theta_raw):
        """The solver's parameters, each held value (a locked centre or width, the
        height of a peak the window cannot see) at its start value.

        Never clipped: every solver fits within (lo, hi), so a value outside them
        is the solver's fault, and is reported (see "outside_bounds") instead of
        being moved onto the bound, where the other parameters would no longer
        be its best fit. Returns theta and which values lie outside."""

        theta = np.asarray(theta_raw if theta_raw is not None else p0, float).copy()
        theta[held] = p0[held]
        tol = 1e-9 * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        return theta, (theta < lo - tol) | (theta > hi + tol)

    def _peaks_at(theta):
        return [
            Peak(*theta[4 * i : 4 * (i + 1)], pk.lock_center, pk.lock_width)
            for i, pk in enumerate(peaks0)
        ]

    # Every solver packs height, centre and width; eta stays where it starts. A
    # held value is no parameter: a locked centre or width, and the height of a
    # peak the window cannot see (fit.bounds.held_heights), each at its p0.
    held = np.array(
        [f for pk, h in zip(peaks0, np.isfinite(held_heights((lo, hi))))
         for f in (pk.lock_center, h, pk.lock_width, False)], bool
    )
    free_mask = ~held
    free_mask[3::4] = False

    best_res, best_theta = _solve(p0, opts, restarts, jitter_pct)

    noise = None
    if sigma_fit is None and noise_choice == "per_spectrum" and best_theta is not None:
        # var_i = s0^2 + g S_i from this fit's residuals, S_i its signal above the
        # baseline; then one refit weighted by 1 / sigma_i, started at this fit.
        theta1, _outside = _finalize(best_theta)
        resid1 = build_residual(x_fit, y_fit, _peaks_at(theta1), mode, _fit_baseline(best_res), "linear", None)
        signal1 = performance.eval_total(
            x_fit, [(h, c, w, e) for c, h, w, e in theta1.reshape(-1, 4)]
        )
        jac1 = jacobian_fd(resid1, theta1)[:, free_mask]
        if joint:
            # the fit shrinks its residuals through the baseline columns too
            jac1 = np.hstack([jac1, B_fit])
        noise = noise_model.estimate_noise_model(resid1(theta1), signal1, jac1)
        # A floor at the data's rounding level is no noise: the model fits the
        # data exactly (noiseless input, which a joint baseline can match to the
        # last digit), and weights of 1e17 would only break the refit.
        rounding = 1e-9 * max(1.0, float(np.max(np.abs(y_fit)))) if y_fit.size else 0.0
        if noise is not None and noise.usable and noise.s0 > rounding:
            sigma_fit = noise.sigma(signal1)
            opts["weights"] = 1.0 / sigma_fit
            weights = noise_weights(y_target, opts["weights"])
            best_res, best_theta = _solve(theta1, opts, 1, 0.0)

    theta, outside = _finalize(best_theta)

    baseline_coef = None
    if joint:
        # The coefficients that best fit the data less the final peaks, with the
        # fit's weights: the solver's own at convergence, and exact after a clip.
        w_final = noise_weights(y_target, opts.get("weights", "none"))
        w_ls = np.ones_like(y_fit) if w_final is None else np.asarray(w_final, float)
        peaks_final = performance.eval_total(x_fit, [(h, c, w, e) for c, h, w, e in theta.reshape(-1, 4)])
        lin_final = baseline_model.LinearPart(B_fit * w_ls[:, None])
        baseline_coef = lin_final.coef((y_fit - peaks_final) * w_ls)
        baseline = jb.evaluate(x, baseline_coef)
        base_fit = baseline[mask]

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

    baseline_cfg = base_cfg
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
            "mask_len=%d window=[%g,%g] bounds=(%g,%g) p0_head=%s p0_tail=%s restarts=%d jitter_pct=%g rmse=%g outside_bounds=%s"
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
                int(np.count_nonzero(outside)),
            )
        )

    result = {
        "peaks_out": peaks_out,
        "rmse": rmse,
        "theta": theta,
        "bounds": (lo, hi),
        # the heights' cap before pack_theta_bounds scales it per peak: a refit
        # (bootstrap) takes it, to fit within the same bounds
        "height_cap": float(opts["max_height"]),
        "p0": p0,
        "mask_len": int(mask.sum()),
        "baseline_params": baseline_params,
        "restarts_used": restarts,
        "jitter_pct": jitter_pct,
        "solver_options": solver_used_opts,
        "fit_ok": bool(best_res.success if best_res is not None else False),
        "skip_reason": None,
        "message": str(getattr(best_res, "message", "") or ""),
        # The noise model's provenance (s0, g, n, method) when one was estimated,
        # and the sigma_i on the fit window that weighted the fit (None: unweighted
        # by a noise model). The uncertainty methods take the same sigma_i.
        "noise_model": noise.as_dict() if noise is not None else None,
        "noise_sigma": None if sigma_fit is None else np.asarray(sigma_fit, float),
        # The baseline: held fixed, or fitted with the peaks (its coefficients
        # and columns over the window).
        "baseline_fit": "joint" if joint else "fixed",
        "baseline_order": joint_order,
        "baseline_coef": baseline_coef,
        "joint_baseline": jb,
    }

    names: list[str] = []
    for i in range(len(peaks_out)):
        names.extend([f"center{i+1}", f"height{i+1}", f"fwhm{i+1}", f"eta{i+1}"])
    locked_mask = ~free_mask
    n_free = int(np.count_nonzero(free_mask))
    # A free parameter on its bound is a flag, not a failure: the fit is kept. So
    # is a held height: the fit says nothing about its peak.
    flagged = free_mask.copy()
    flagged[1::4] = True
    bound_hit = _params_on_bounds(theta, lo, hi, ~flagged, names)
    outside_bounds = [n for n, out in zip(names, outside) if out]
    result.update(
        {
            "param_names": names,
            "locked_mask": locked_mask,
            "flags": ["bound_hit"] if bound_hit else [],
            "bound_hit": bound_hit,
            "outside_bounds": outside_bounds,
        }
    )
    if outside_bounds:
        result["fit_ok"] = False
        result["message"] = (
            f"the {solver_name} solver returned values outside the fit's bounds: {', '.join(outside_bounds)}"
        )

    if return_jacobian:
        r0 = resid_fn(theta)
        J = jacobian_fd(resid_fn, theta)
        # Degrees of freedom count the fitted parameters only (the joint
        # baseline's coefficients among them); none are left at <= 0.
        n_fitted = n_free + n_baseline
        dof = r0.size - n_fitted if r0.size > n_fitted else float("nan")

        result.update(
            {
                "jacobian": J,
                "residual": r0,
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
    result.update({"baseline": baseline, "base_all": base_all, "x": x, "mode": mode})
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
                "vp_unpack_cw": lambda theta_nl, struct=struct, peaks=peaks_out: vp_unpack_centers_widths(theta_nl, struct, peaks),
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
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    opts["max_height"] = height_cap(y_target, opts)

    theta_full0, bounds_full = pack_theta_bounds(peaks, x, opts)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(opts.get("min_fwhm", 1e-6)), 2.0 * dx_med)
    theta0, bounds, x_scale, indices = modern._to_solver_vectors(
        theta_full0, bounds_full, peaks, fwhm_min
    )
    weights = noise_weights(y_target, opts.get("weights", "none"))
    resid_jac = build_residual_jac(
        x, y, peaks, mode, baseline, weights, held_heights=held_heights(bounds_full)
    )
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
    _, (_lo, hi) = pack_theta_bounds(peaks, x, opts)
    h = baseline_model.nonneg_heights(Aw, bw, np.asarray(hi, float)[1::4])
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
    opts["max_height"] = height_cap(y_target, opts)
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
    opts["max_height"] = height_cap(y_target, opts)
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


