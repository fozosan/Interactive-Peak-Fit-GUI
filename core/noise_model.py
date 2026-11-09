"""Per-spectrum count-noise model: var_i = s0^2 + g * S_i.

S_i is the fitted signal above the baseline at point i, in the data's units
(counts). s0 is the noise floor, which holds the read noise and the baseline's
own shot noise; g is the variance added per count of signal. With g = 0 the
model is one constant sigma, the noise every uncertainty method assumed before
(game plan 1.5, decision D1: estimated for each measurement).

The model is estimated from the residuals r of a first fit, through their 4th
differences, d = C r with C the normalised stencil (1, -4, 6, -4, 1) / sqrt(70):

- A smooth misfit cancels in d. On the user's spectra the ALS baseline bends
  into the lines and eta is held at 0.5, and the plain squared residuals read
  that misfit as signal-dependent noise: g came out 8 to 19 times the
  detector's on six of the MoS2 spectra (PR D probes, 2026-11-08). The 4th
  differences give 0.033 to 0.065 on the strong MoS2 and Si lines, where the
  review measured 0.0451 with the same filter. ``order=0`` uses the residuals
  themselves.
- The fit shrinks its residuals, most at a narrow line's top, where the
  leverage is near 1. The regressors carry that exactly: with H the fit's hat
  matrix and A = C (I - H), E[d_i^2] = s0^2 (M 1)_i + g (M S)_i, M = A o A. With
  no Jacobian, H = 0.
- The shape kappa = g / s0^2 comes from a robust regression of d_i^2 on those
  two regressors: Gaussian maximum likelihood (gamma-GLM IRLS), each d_i^2 capped
  at ``cap`` times its model variance so a cosmic ray cannot pull it, and both
  coefficients >= 0.
- The floor s0 is 1.4826 x MAD (cafm's ``robust_sigma``) of the d_i standardised
  by that shape, d_i / sqrt((M 1)_i + kappa (M S)_i); g = kappa s0^2.

The filter assumes uncorrelated noise. The review measured a lag-1 correlation
of -0.08 on this detector, which raises the 4th-difference variance by 13 %, so
s0 and g read about 6 % and 13 % above the point variance there. The review's
pooled model, var = 0.724 + 0.0451 * S (S above the 502.5-count bias), was
measured the same way. The intervals do not depend on it: the asymptotic
covariance is scaled by the weighted chi^2 / dof, the bootstrap resamples the
residuals at their own size and the Bayesian samples the scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erf, exp, isfinite, pi, sqrt
from typing import Optional

import numpy as np

MAD_TO_SIGMA = 1.4826  # 1 / Phi^-1(0.75): MAD -> sigma for a normal distribution
# A filtered residual counts at most CAP times its model variance.
CAP = 9.0
# The difference order of the filter: 4 cancels a locally cubic misfit.
ORDER = 4


def robust_sigma(arr) -> float:
    """1.4826 x MAD of the finite values, as cafm's ``robust_sigma``.

    Empty input gives 0.0. When MAD is 0 (more than half the values identical)
    the standard deviation is returned instead.
    """

    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    sig = MAD_TO_SIGMA * float(np.median(np.abs(a - np.median(a))))
    if sig <= 0.0:
        sig = float(np.std(a))
    return sig


@dataclass(frozen=True)
class NoiseModel:
    """var_i = s0^2 + g * S_i, with S_i the fitted signal above the baseline."""

    s0: float
    g: float
    n: int
    method: str = f"fit_residuals_d{ORDER}"

    def variance(self, signal) -> np.ndarray:
        S = np.maximum(np.asarray(signal, dtype=float), 0.0)
        return self.s0 * self.s0 + self.g * S

    def sigma(self, signal) -> np.ndarray:
        return np.sqrt(self.variance(signal))

    @property
    def usable(self) -> bool:
        """A model can weight a fit only if its floor is positive."""

        return bool(np.isfinite(self.s0) and self.s0 > 0.0 and np.isfinite(self.g) and self.g >= 0.0)

    def as_dict(self) -> dict:
        return {"s0": float(self.s0), "g": float(self.g), "n": int(self.n), "method": str(self.method)}


def difference_stencil(order: int) -> np.ndarray:
    """(-1)^k binom(order, k), scaled to unit norm: d = C r has the variance of r."""

    c = np.array([(-1) ** k * comb(int(order), k) for k in range(int(order) + 1)], float)
    return c / np.linalg.norm(c)


def variance_regressors(jacobian, signal, order: int = ORDER) -> np.ndarray:
    """[(M 1), (M S)] with M = (C (I - H)) o (C (I - H)), one row per filtered point.

    E[d_i^2] = s0^2 (M 1)_i + g (M S)_i for d = C r, r = (I - H) e the residuals of
    a least-squares fit with hat matrix H = J J^+ and e ~ N(0, s0^2 + g S).
    Computed from C - (C J) J^+ without forming the n x n matrices.
    """

    S = np.asarray(signal, dtype=float).ravel()
    n = S.size
    c = difference_stencil(order)
    L = c.size
    m = n - L + 1
    J = np.zeros((n, 0)) if jacobian is None else np.asarray(jacobian, dtype=float).reshape(n, -1)
    Jp = np.linalg.pinv(J) if J.shape[1] else np.zeros((0, n))
    U = sum(c[k] * J[k : k + m] for k in range(L))  # C J, m x p
    out = []
    for w in (np.ones(n), S):
        t1 = sum(c[k] ** 2 * w[k : k + m] for k in range(L))
        # sum_k c_k w_(i+k) (C J J^+)_(i, i+k)
        t2 = sum(c[k] * w[k : k + m] * np.einsum("ip,pi->i", U, Jp[:, k : k + m]) for k in range(L))
        G = (Jp * w[None, :]) @ Jp.T
        t3 = np.einsum("ip,pq,iq->i", U, G, U)
        out.append(t1 - 2.0 * t2 + t3)
    return np.column_stack(out)


def _capped_chi2_mean(cap: float) -> float:
    """E[min(Z^2, cap)] for a standard normal Z."""

    if not isfinite(cap):
        return 1.0
    rc = sqrt(cap)
    p_in = erf(rc / sqrt(2.0))
    pdf = exp(-0.5 * cap) / sqrt(2.0 * pi)
    return p_in - 2.0 * rc * pdf + cap * (1.0 - p_in)


def _variance_regression(q: np.ndarray, X: np.ndarray, cap: float) -> tuple[float, float]:
    """(a, g) >= 0 with E[q] = a X[:, 0] + g X[:, 1]: capped gamma-GLM IRLS.

    The regression is homogeneous in q, so it runs on q in units of its mean:
    residuals at the rounding level (a model that fits noiseless data exactly,
    as a joint baseline can) then neither underflow nor overflow.
    """

    scale = float(np.mean(q / X[:, 0]))
    if not (np.isfinite(scale) and scale > 0.0):
        return 0.0, 0.0
    q = q / scale
    cap_mean = _capped_chi2_mean(cap)
    a = max(MAD_TO_SIGMA**2 * float(np.median(q / X[:, 0])), 1e-12)
    g = 0.0
    slope = np.any(X[:, 1] > 0.0)
    for _ in range(300):
        v = np.maximum(a * X[:, 0] + g * X[:, 1], 1e-300)
        qc = np.minimum(q, cap * v) / cap_mean
        sw = 1.0 / v
        a_new = g_new = -1.0
        if slope:
            coef = np.linalg.lstsq(X * sw[:, None], qc * sw, rcond=None)[0]
            a_new, g_new = float(coef[0]), float(coef[1])
        if g_new < 0.0 or a_new <= 0.0:
            # The boundary, or no signal at all: one constant variance.
            g_new = 0.0
            w = sw * sw
            a_new = float(np.sum(w * qc * X[:, 0]) / np.sum(w * X[:, 0] ** 2))
        done = abs(a_new - a) <= 1e-10 * a and abs(g_new - g) <= 1e-10 * max(g_new, 1e-300)
        a, g = a_new, g_new
        if done:
            break
    return a * scale, g * scale


def estimate_noise_model(
    residual, signal, jacobian=None, *, order: int = ORDER, cap: float = CAP
) -> Optional[NoiseModel]:
    """Estimate var_i = s0^2 + g * S_i from a fit's residuals.

    ``residual`` and ``signal`` are given at the fitted points, in order along x;
    ``signal`` is the fitted model above the baseline. ``jacobian`` is the fit's
    Jacobian over its free parameters (one row per point), which carries the
    shrinkage of fitted residuals; without it none is assumed. ``order`` is the
    difference order of the filter (0: the residuals themselves). Returns None
    when a value is not finite or fewer than 3 filtered values are left.
    """

    r = np.asarray(residual, dtype=float).ravel()
    S = np.maximum(np.asarray(signal, dtype=float).ravel(), 0.0)
    if r.size != S.size:
        raise ValueError(f"residual has {r.size} points, signal {S.size}")
    if jacobian is not None and np.asarray(jacobian).shape[0] != r.size:
        raise ValueError(f"jacobian has {np.asarray(jacobian).shape[0]} rows, residual {r.size} points")
    if r.size - int(order) < 3 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(S))):
        return None
    c = difference_stencil(order)
    m = r.size - c.size + 1
    # About the median, the centre MAD uses: a baseline offset is not noise.
    rc = r - np.median(r)
    d = sum(c[k] * rc[k : k + m] for k in range(c.size))
    X = variance_regressors(jacobian, S, order)
    # A filtered value the fit pins (its expected variance ~0) carries nothing.
    keep = X[:, 0] > 1e-6 * float(np.max(X[:, 0]))
    d, X = d[keep], X[keep]
    method = f"fit_residuals_d{int(order)}"
    if d.size < 3 or robust_sigma(d / np.sqrt(X[:, 0])) <= 0.0:
        return NoiseModel(s0=0.0, g=0.0, n=int(r.size), method=method)
    a, g = _variance_regression(d * d, X, float(cap))
    kappa = g / a if a > 0.0 else 0.0
    s0 = robust_sigma(d / np.sqrt(X[:, 0] + kappa * X[:, 1]))
    return NoiseModel(s0=s0, g=kappa * s0 * s0, n=int(r.size), method=method)
