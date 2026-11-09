"""The baseline as part of the model: peaks plus a baseline (game plan 1.6, decision D2).

One model serves both paths:

- ``baseline_fit = "fixed"`` (the core default, and the interactive path) holds a
  precomputed baseline (ALS or polynomial) fixed. It is the special case with no
  free baseline columns.
- ``baseline_fit = "joint"`` (the batch default) fits the baseline with the peaks:
  a Legendre series of order ``baseline_order`` over the fit window, whose
  coefficients are free linear parameters. The variable-projection solvers solve
  them beside the peak heights; the solvers that move every parameter (TRF,
  Classic) carry them as extra parameters. Every uncertainty method then
  includes them.

The columns are centred and scaled: P_j(u), u = 2 (x - lo) / (hi - lo) - 1 over
the window's points, so they do not depend on the x units or on where the
window sits, and they are close to orthogonal on a uniform grid: cond 2.6 for
order 3 on 340-440 cm-1, where raw cm-1 monomials reach 2e11 (prior art:
``fit-affines-on-centered-coordinates``). Outside the window the baseline
stays at its edge values.

The default order 3 was measured (PR E probes, 2026-11-08). On the 5 s MoS2
spectrum the background measured over 300-480 cm-1 bends 0.64 counts away from
a straight line inside 340-440 and 0.31 from a parabola; a cubic follows it to
0.07. On synthetic spectra with that background, the measured noise and the
review's truths (weak MoS2, strong MoS2 and Si, 200 replicates each), the
largest centre or FWHM bias was 0.70 true sd with a linear baseline, 0.40 with
a quadratic, 0.10 with a cubic and 0.10 with a quartic, which widened the Si
FWHM's sd by 13 % over the cubic. On the real MoS2 spectra BIC prefers order 0
or 1, and the cubic costs 1 to 4 % in SE against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as _legendre
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

BASELINE_FITS = ("fixed", "joint")
DEFAULT_ORDER = 3
COEF_PREFIX = "baseline_c"


def baseline_fit_choice(cfg: Optional[dict]) -> str:
    """``cfg["baseline_fit"]``: "fixed" (default) or "joint"; anything else is an error."""

    raw = (cfg or {}).get("baseline_fit", "fixed")
    if raw is None:
        return "fixed"
    choice = str(raw).strip().lower()
    if choice == "":
        return "fixed"
    if choice not in BASELINE_FITS:
        raise ValueError(f"Unknown baseline_fit {raw!r}; expected one of {BASELINE_FITS}")
    return choice


def baseline_order(cfg: Optional[dict]) -> int:
    """``cfg["baseline_order"]``: the joint baseline's Legendre order, a whole number >= 0."""

    raw = (cfg or {}).get("baseline_order", DEFAULT_ORDER)
    if raw is None:
        return DEFAULT_ORDER
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"baseline_order must be a whole number >= 0 (got {raw!r})") from None
    if not np.isfinite(val) or val < 0 or val != int(val):
        raise ValueError(f"baseline_order must be a whole number >= 0 (got {raw!r})")
    return int(val)


@dataclass(frozen=True)
class JointBaseline:
    """Legendre columns of order ``order`` over [lo, hi], the fit window's points."""

    order: int
    lo: float
    hi: float

    @classmethod
    def over(cls, x_window, order: int) -> "JointBaseline":
        xw = np.asarray(x_window, dtype=float).ravel()
        if xw.size == 0:
            raise ValueError("the joint baseline needs at least one point in the fit window")
        lo, hi = float(np.min(xw)), float(np.max(xw))
        return cls(order=int(order), lo=lo, hi=hi)

    @property
    def n_coef(self) -> int:
        return self.order + 1

    def u(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        span = self.hi - self.lo
        if span <= 0.0:
            return np.zeros_like(x)
        return np.clip(2.0 * (x - self.lo) / span - 1.0, -1.0, 1.0)

    def basis(self, x) -> np.ndarray:
        """The columns P_0(u) .. P_order(u), one row per point of ``x``."""

        return _legendre.legvander(self.u(x), self.order)

    def evaluate(self, x, coef) -> np.ndarray:
        return self.basis(x) @ np.asarray(coef, dtype=float)

    def names(self) -> List[str]:
        return [f"{COEF_PREFIX}{j}" for j in range(self.n_coef)]

    def as_dict(self) -> Dict[str, Any]:
        return {"fit": "joint", "order": self.order, "basis": "legendre", "x_lo": self.lo, "x_hi": self.hi}


class LinearPart:
    """The model's free linear columns beside the peaks: weighted baseline columns ``Bw``.

    ``solve`` returns the heights h >= 0 and the coefficients c (free) that
    minimise ||Aw h + Bw c - bw||. It is exact: with Bw = U R, c is projected
    out, h = NNLS((I - U U^T) Aw, (I - U U^T) bw), then c = R^-1 U^T (bw - Aw h).
    ``project`` applies I - U U^T, the step that turns a Jacobian at fixed
    linear parameters into the variable-projection one. With no columns both
    reduce to plain NNLS and the identity.
    """

    def __init__(self, Bw: Optional[np.ndarray]):
        if Bw is None or np.asarray(Bw).size == 0:
            self.U = None
            self.R = None
            self.m = 0
        else:
            Bw = np.asarray(Bw, dtype=float)
            self.U, self.R = np.linalg.qr(Bw)
            self.m = int(Bw.shape[1])

    def project(self, M: np.ndarray) -> np.ndarray:
        if self.U is None:
            return M
        M = np.asarray(M, dtype=float)
        return M - self.U @ (self.U.T @ M)

    def coef(self, rhs: np.ndarray) -> np.ndarray:
        """The coefficients that best fit ``rhs`` (weighted, in the columns' units)."""

        if self.U is None:
            return np.zeros(0)
        return solve_triangular(self.R, self.U.T @ np.asarray(rhs, dtype=float))

    def solve(self, Aw: np.ndarray, bw: np.ndarray, max_height: float = np.inf) -> Tuple[np.ndarray, np.ndarray]:
        Aw = np.asarray(Aw, dtype=float)
        bw = np.asarray(bw, dtype=float)
        if Aw.shape[1]:
            h, _ = nnls(self.project(Aw), self.project(bw))
            h = np.minimum(h, max_height)
        else:
            h = np.zeros(0)
        return h, self.coef(bw - Aw @ h)


def joint_from_options(options: Optional[dict], n: int) -> Optional[np.ndarray]:
    """The unweighted baseline columns a solver gets in ``options["baseline_basis"]``."""

    B = (options or {}).get("baseline_basis")
    if B is None:
        return None
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != int(n):
        raise ValueError(f"baseline_basis has shape {B.shape} for {int(n)} points")
    return B


def uncertainty_problem(res: dict, x_fit, y_fit) -> Dict[str, Any]:
    """A joint fit's uncertainty problem on the window, the coefficients as parameters.

    ``res`` is ``run_fit_consistent``'s result (``return_jacobian=True``); x_fit and
    y_fit are the window's points and data. Returns the parameter vector
    [theta, coef], the residual (model - data) and predictor on the window
    (``predict_batch``: the same model for rows of parameter vectors), the
    Jacobian with the baseline columns appended, the lock mask, bounds and
    names extended for the coefficients (free, unbounded), and the number of
    peak parameters in front.
    """

    jb: JointBaseline = res["joint_baseline"]
    theta = np.asarray(res["theta"], dtype=float)
    coef = np.asarray(res["baseline_coef"], dtype=float)
    xf = np.asarray(x_fit, dtype=float)
    yf = np.asarray(y_fit, dtype=float)
    B = jb.basis(xf)
    P = theta.size
    m = coef.size
    from . import models  # local: models imports nothing from here

    def predict(th) -> np.ndarray:
        t = np.asarray(th, dtype=float).ravel()
        out = B @ t[P:]
        for i in range(P // 4):
            c, h, w, e = t[4 * i : 4 * i + 4]
            out = out + models.pseudo_voigt(xf, h, c, w, e)
        return out

    def predict_batch(rows) -> np.ndarray:
        T = np.atleast_2d(np.asarray(rows, dtype=float))
        return models.pseudo_voigt_rows(xf, T[:, :P]) + T[:, P:] @ B.T

    def residual_fn(th) -> np.ndarray:
        return predict(th) - yf

    J = np.asarray(res["jacobian"], dtype=float)
    lo, hi = res["bounds"]
    return {
        "theta": np.r_[theta, coef],
        "predict": predict,
        "predict_batch": predict_batch,
        "residual_fn": residual_fn,
        "jacobian": np.hstack([J, B]),
        "locked_mask": np.r_[np.asarray(res["locked_mask"], bool), np.zeros(m, bool)],
        "bounds": (np.r_[np.asarray(lo, float), np.full(m, -np.inf)], np.r_[np.asarray(hi, float), np.full(m, np.inf)]),
        "param_names": list(res["param_names"]) + jb.names(),
        "n_peak_params": P,
        "basis": B,
    }
