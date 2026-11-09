"""The batch's Bayesian problem, built by hand for the Bayesian unit tests.

Strong MoS2 (E2g 68.7 and A1g 92.3 counts) over a sloped background in the
340-440 cm-1 window, with the measured count noise as the noise model's sigma_i
(var = 0.724 + 0.0451 * S, see ``tests/synthetic_raman.py``) and a cubic joint
baseline (game plan 1.6): theta is the peaks' (center, height, fwhm, eta) and
then the four Legendre coefficients, as ``batch.runner`` hands it to
``bayesian_ci``. The bounds are the solver's, so the heights carry its 1e12
sentinel. theta_hat is the weighted least-squares fit, solved here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from core.models import pseudo_voigt
from tests.synthetic_raman import BIAS, MOS2_LINES, NOISE_FLOOR, NOISE_GAIN

N_POINTS = 135
X = np.linspace(340.0, 440.0, N_POINTS)
U = 2.0 * (X - X[0]) / (X[-1] - X[0]) - 1.0
BASIS = np.polynomial.legendre.legvander(U, 3)
BASE_COEF = np.array([504.9, 0.85, 0.0, 0.0])


def model(theta, x=X, basis=BASIS):
    t = np.asarray(theta, float).ravel()
    n_pk = (t.size - basis.shape[1]) // 4
    out = basis @ t[4 * n_pk:]
    for i in range(n_pk):
        c, h, w, e = t[4 * i: 4 * i + 4]
        out = out + pseudo_voigt(x, h, c, w, e)
    return out


def mos2_problem(seed: int = 0, *, lines=MOS2_LINES):
    """kwargs for ``bayesian_ci`` (without the chain knobs) and the pieces they came from."""

    truth = np.r_[np.ravel([[c, h, w, e] for c, h, w, e in lines]), BASE_COEF]
    mu = model(truth)
    sigma = np.sqrt(NOISE_FLOOR + NOISE_GAIN * np.maximum(mu - BIAS, 0.0))
    y = mu + sigma * np.random.default_rng(seed).standard_normal(X.size)
    n_pk = len(lines)
    free = np.ones(truth.size, bool)
    free[3:4 * n_pk:4] = False  # eta: no solver fits it

    def wres(p):
        t = truth.copy()
        t[free] = p
        return (y - model(t)) / sigma

    fit = least_squares(wres, truth[free], x_scale="jac")
    theta_hat = truth.copy()
    theta_hat[free] = fit.x
    locked = np.zeros(truth.size, bool)
    locked[3:4 * n_pk:4] = True
    lo = np.r_[np.tile([330.0, 0.0, 1.48, 0.0], n_pk), np.full(4, -np.inf)]
    hi = np.r_[np.tile([450.0, 1e12, 50.0, 1.0], n_pk), np.full(4, np.inf)]
    kwargs = dict(
        theta_hat=theta_hat,
        model=model,
        predict_full=model,
        x_all=X,
        y_all=y,
        bounds=(lo, hi),
        locked_mask=locked,
        param_names=[f"{k}{i + 1}" for i in range(n_pk) for k in ("center", "height", "fwhm", "eta")]
        + [f"baseline_c{j}" for j in range(4)],
        sigma=sigma,
        fit_ctx={"baseline_basis": BASIS, "alpha": 0.05},
        return_band=False,
    )
    return kwargs, {"truth": truth, "sigma": sigma, "y": y, "n_pk": n_pk}


def recording_sampler(monkeypatch):
    """Swap in an ``emcee.EnsembleSampler`` that keeps its constructor keywords and the
    walkers' start. Returns the list the samplers are appended to."""

    import emcee

    made = []

    class Recorder(emcee.EnsembleSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_kwargs = dict(kwargs)
            self.start = None
            made.append(self)

        def run_mcmc(self, initial_state, nsteps, **kwargs):
            if self.start is None:
                self.start = np.array(getattr(initial_state, "coords", initial_state), float)
            return super().run_mcmc(initial_state, nsteps, **kwargs)

    monkeypatch.setattr(emcee, "EnsembleSampler", Recorder)
    return made


def sampled_index(n_pk: int):
    """theta's indices in the order the sampler holds them: each peak's center, height, fwhm."""

    return [4 * i + k for i in range(n_pk) for k in (0, 1, 2)]


def asymptotic_se(kwargs) -> np.ndarray:
    """SE of the sampled peak parameters, by hand: s^2 (J^T J)^-1 of the weighted residual
    with the baseline columns projected out, s^2 = RSS / (n - m - p)."""

    th = np.asarray(kwargs["theta_hat"], float)
    y = kwargs["y_all"]
    sig = kwargs["sigma"]
    Bw = kwargs["fit_ctx"]["baseline_basis"] / sig[:, None]
    Q, _ = np.linalg.qr(Bw)
    n_pk = (th.size - Bw.shape[1]) // 4

    def proj(v):
        return v - Q @ (Q.T @ v)

    idx = sampled_index(n_pk)
    r0 = proj((y - model(th)) / sig)
    J = np.empty((y.size, len(idx)))
    for k, j in enumerate(idx):
        h = 1e-6 * max(abs(th[j]), 1.0)
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        J[:, k] = proj((model(tp) - model(tm)) / (2.0 * h) / sig)
    s2 = float(r0 @ r0) / (y.size - Bw.shape[1] - len(idx))
    return np.sqrt(np.diag(s2 * np.linalg.inv(J.T @ J)))
