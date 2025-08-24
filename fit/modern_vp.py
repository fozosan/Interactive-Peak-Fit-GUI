"""Variable projection solver using NNLS for pseudo-Voigt peaks."""
from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import nnls

from core.models import pv_design_matrix, pv_sum_with_jac
from core.peaks import Peak
from .bounds import pack_theta_bounds
from .utils import mad_sigma, robust_cost
from . import step_engine


class SolveResult(TypedDict):
    ok: bool
    theta: np.ndarray
    message: str
    cost: float
    jac: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    meta: dict


def _to_solver_vectors(theta0: np.ndarray, bounds, peaks, fwhm_min: float):
    lb, ub = bounds
    theta_list = []
    lb_list = []
    ub_list = []
    x_scale = []
    indices = []
    map_list: list[tuple[int, str]] = []
    for i, p in enumerate(peaks):
        if not p.lock_center:
            c = theta0[4 * i + 0]
            theta_list.append(c)
            lb_list.append(lb[4 * i + 0])
            ub_list.append(ub[4 * i + 0])
            x_scale.append(max(theta0[4 * i + 2], fwhm_min))
            indices.append(4 * i + 0)
            map_list.append((i, "c"))
        if not p.lock_width:
            w = theta0[4 * i + 2]
            theta_list.append(w)
            lb_list.append(lb[4 * i + 2])
            ub_list.append(ub[4 * i + 2])
            x_scale.append(max(w, fwhm_min))
            indices.append(4 * i + 2)
            map_list.append((i, "f"))
    return (
        np.asarray(theta_list, dtype=float),
        (np.asarray(lb_list, dtype=float), np.asarray(ub_list, dtype=float)),
        np.asarray(x_scale, dtype=float),
        np.asarray(indices, dtype=int),
        map_list,
    )


def solve(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    mode: str,
    baseline: np.ndarray | None,
    options: dict,
) -> SolveResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float) if baseline is not None else None

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    f_scale_opt = float(options.get("f_scale", 0.0))
    maxfev = int(options.get("maxfev", 100))
    lambda_c = float(options.get("lambda_c", 0.0))
    lambda_w = float(options.get("lambda_w", 0.0))

    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(np.abs(y), 1e-12, None)

    options = options.copy()
    base = baseline if baseline is not None else 0.0
    y_target = y - base
    p95 = float(np.percentile(np.abs(y_target), 95)) if y_target.size else 1.0
    max_height_factor = float(options.get("max_height_factor", np.inf))
    options["max_height"] = max_height_factor * p95
    options["max_fwhm"] = options.get("max_fwhm", 0.5 * (x.max() - x.min()))

    theta0_full, bounds_full = pack_theta_bounds(peaks, x, options)
    dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm_min = max(float(options.get("min_fwhm", 1e-6)), 2.0 * dx_med)

    all_locked = all(p.lock_center and p.lock_width for p in peaks)

    if all_locked:
        A = pv_design_matrix(x, peaks)
        b = y_target
        if weights is not None:
            Aw = A * weights[:, None]
            bw = b * weights
        else:
            Aw = A
            bw = b
        h, _ = nnls(Aw, bw)
        h = np.minimum(h, options["max_height"])
        model = A @ h
        r = model - b
        if weights is not None:
            r = r * weights
        sigma = mad_sigma(r)
        fs = f_scale_opt if f_scale_opt > 0 else max(sigma, 1e-12)
        cost = robust_cost(r, loss, fs)
        theta_full = theta0_full.copy()
        for i, val in enumerate(h):
            theta_full[4 * i + 1] = val
        return SolveResult(
            ok=True,
            theta=theta_full,
            message="linear",
            cost=cost,
            jac=Aw if weights is not None else A,
            cov=None,
            meta={"nfev": 1, "sigma": sigma, "f_scale": fs},
        )

    theta0, bounds, x_scale, indices, mapping = _to_solver_vectors(theta0_full, bounds_full, peaks, fwhm_min)
    c0 = np.array([p.center for p in peaks], dtype=float)
    f0 = np.array([p.fwhm for p in peaks], dtype=float)

    theta = theta0.copy()
    lb, ub = bounds
    n = len(theta)

    def unpack(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = c0.copy()
        f = f0.copy()
        j = 0
        for (pk_idx, kind) in mapping:
            if kind == "c":
                c[pk_idx] = t[j]
            else:
                f[pk_idx] = t[j]
            j += 1
        return c, f

    sigma = 1.0
    fs = f_scale_opt if f_scale_opt > 0 else 1.0

    backtracked = False
    for nfev in range(1, maxfev + 1):
        c, f = unpack(theta)
        pk_iter = [Peak(c[i], 1.0, f[i], peaks[i].eta) for i in range(len(peaks))]
        A = pv_design_matrix(x, pk_iter)
        b = y_target
        if weights is not None:
            Aw = A * weights[:, None]
            bw = b * weights
        else:
            Aw = A
            bw = b
        h, _ = nnls(Aw, bw)
        h = np.minimum(h, options["max_height"])
        model = A @ h
        r = model - b
        if weights is not None:
            r = r * weights

        if loss != "linear" and f_scale_opt <= 0:
            sigma = mad_sigma(r)
            fs = max(sigma, 1e-12)
        else:
            fs = f_scale_opt if f_scale_opt > 0 else sigma
        cost = robust_cost(r, loss, fs)

        y_model, dh, dc, df = pv_sum_with_jac(x, c, h, f, np.array([p.eta for p in peaks], dtype=float))
        cols = []
        for (pk_idx, kind) in mapping:
            if kind == "c":
                cols.append(dc[:, pk_idx])
            else:
                cols.append(df[:, pk_idx])
        J = np.column_stack(cols) if cols else np.zeros((x.size, 0))
        if weights is not None:
            J = J * weights[:, None]

        tether_rows = []
        tether_vals = []
        if lambda_c > 0 or lambda_w > 0:
            for j, (pk_idx, kind) in enumerate(mapping):
                if kind == "c" and lambda_c > 0:
                    scale = np.sqrt(lambda_c) / max(f0[pk_idx], fwhm_min)
                    tether_vals.append(scale * (c[pk_idx] - c0[pk_idx]))
                    row = np.zeros(J.shape[1])
                    row[j] = scale
                    tether_rows.append(row)
                elif kind == "f" and lambda_w > 0:
                    scale = np.sqrt(lambda_w)
                    tether_vals.append(scale * np.log(f[pk_idx] / max(f0[pk_idx], fwhm_min)))
                    row = np.zeros(J.shape[1])
                    row[j] = scale / f[pk_idx]
                    tether_rows.append(row)
        if tether_vals:
            J = np.vstack([J, np.asarray(tether_rows)])
            r = np.concatenate([r, np.asarray(tether_vals)])

        # Gauss-Newton step with scaling
        if J.size == 0:
            break
        J_scaled = J / x_scale
        try:
            step_scaled = np.linalg.lstsq(J_scaled, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        step = step_scaled * x_scale

        # Backtracking
        theta_new = np.minimum(np.maximum(theta + step, lb), ub)
        c_new, f_new = unpack(theta_new)
        pk_new = [Peak(c_new[i], 1.0, f_new[i], peaks[i].eta) for i in range(len(peaks))]
        A_new = pv_design_matrix(x, pk_new)
        if weights is not None:
            Aw_new = A_new * weights[:, None]
            bw = b * weights
        else:
            Aw_new = A_new
            bw = b
        h_new, _ = nnls(Aw_new, bw)
        h_new = np.minimum(h_new, options["max_height"])
        r_new = A_new @ h_new - b
        if weights is not None:
            r_new = r_new * weights
        cost_new = robust_cost(r_new, loss, fs)
        bt = False
        while cost_new > cost and np.linalg.norm(step) > 1e-8 * (1 + np.linalg.norm(theta)):
            step *= 0.5
            theta_new = np.minimum(np.maximum(theta + step, lb), ub)
            c_new, f_new = unpack(theta_new)
            pk_new = [Peak(c_new[i], 1.0, f_new[i], peaks[i].eta) for i in range(len(peaks))]
            A_new = pv_design_matrix(x, pk_new)
            if weights is not None:
                Aw_new = A_new * weights[:, None]
            else:
                Aw_new = A_new
            h_new, _ = nnls(Aw_new, bw)
            h_new = np.minimum(h_new, options["max_height"])
            r_new = A_new @ h_new - b
            if weights is not None:
                r_new = r_new * weights
            cost_new = robust_cost(r_new, loss, fs)
            bt = True

        theta = theta_new
        backtracked = backtracked or bt
        if np.linalg.norm(step) <= 1e-8 * (1 + np.linalg.norm(theta)):
            break

    # final solve with accepted parameters
    c, f = unpack(theta)
    pk_final = [Peak(c[i], 1.0, f[i], peaks[i].eta) for i in range(len(peaks))]
    A = pv_design_matrix(x, pk_final)
    if weights is not None:
        Aw = A * weights[:, None]
        bw = b * weights
    else:
        Aw = A
        bw = b
    h, _ = nnls(Aw, bw)
    h = np.minimum(h, options["max_height"])
    model = A @ h
    r = model - b
    if weights is not None:
        r = r * weights
    if lambda_c > 0 or lambda_w > 0:
        tether_rows = []
        tether_vals = []
        for j, (pk_idx, kind) in enumerate(mapping):
            if kind == "c" and lambda_c > 0:
                scale = np.sqrt(lambda_c) / max(f0[pk_idx], fwhm_min)
                tether_vals.append(scale * (c[pk_idx] - c0[pk_idx]))
                row = np.zeros(J.shape[1])
                row[j] = scale
                tether_rows.append(row)
            elif kind == "f" and lambda_w > 0:
                scale = np.sqrt(lambda_w)
                tether_vals.append(scale * np.log(f[pk_idx] / max(f0[pk_idx], fwhm_min)))
                row = np.zeros(J.shape[1])
                row[j] = scale / f[pk_idx]
                tether_rows.append(row)
        if tether_vals:
            J = np.vstack([J, np.asarray(tether_rows)])
            r = np.concatenate([r, np.asarray(tether_vals)])

    cost = robust_cost(r, loss, fs)
    sigma = mad_sigma(r)

    # Jacobian at final
    y_model, dh, dc, df = pv_sum_with_jac(x, c, h, f, np.array([p.eta for p in peaks], dtype=float))
    cols = []
    for (pk_idx, kind) in mapping:
        if kind == "c":
            cols.append(dc[:, pk_idx])
        else:
            cols.append(df[:, pk_idx])
    J = np.column_stack(cols) if cols else np.zeros((x.size, 0))
    if weights is not None:
        J = J * weights[:, None]

    theta_full = theta0_full.copy()
    theta_full[indices] = theta
    for i, val in enumerate(h):
        theta_full[4 * i + 1] = val

    cov = None
    if J.size:
        try:
            cov = np.linalg.pinv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = None

    return SolveResult(
        ok=True,
        theta=theta_full,
        message="vp",
        cost=cost,
        jac=J,
        cov=cov,
        meta={"nfev": nfev, "sigma": sigma, "f_scale": fs, "backtracked": backtracked},
    )


def iterate(state: dict) -> dict:
    """Single iteration for the variable projection solver.

    This helper mirrors the behaviour of :func:`solve` but performs only a
    single Gauss–Newton/LM step by delegating to :func:`step_engine.step_once`.
    It accepts and returns a ``state`` dictionary similar to the other solver
    backends.
    """

    x = state["x_fit"]
    y = state["y_fit"]
    peaks = state["peaks"]
    mode = state.get("mode", "subtract")
    baseline = state.get("baseline")
    options = state.get("options", {})

    loss = options.get("loss", "linear")
    weight_mode = options.get("weights", "none")
    weights = None
    if weight_mode == "poisson":
        weights = 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    elif weight_mode == "inv_y":
        weights = 1.0 / np.clip(np.abs(y), 1e-12, None)

    _, bounds = pack_theta_bounds(peaks, x, options)

    theta, cost = step_engine.step_once(
        x,
        y,
        peaks,
        mode,
        baseline,
        loss=loss,
        weights=weights,
        damping=state.get("lambda", 0.0),
        trust_radius=state.get("trust_radius", np.inf),
        bounds=bounds,
        f_scale=options.get("f_scale", 1.0),
    )

    state["theta"] = theta
    state["cost"] = cost
    return state
