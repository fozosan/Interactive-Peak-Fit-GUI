from __future__ import annotations
import numpy as np

def rhat_split(chains: np.ndarray) -> float:
    """Split R-hat with each walker taken as a chain: a display heuristic.

    chains: shape (n_chain, n_draw, n_param). R-hat compares chains run
    independently from dispersed starts. The walkers of one affine-invariant
    ensemble (emcee) are neither: each move uses the other walkers, and they
    start in one tight ball. So this number cannot show an ensemble that is
    stuck or has missed a mode, and at short chains it reads high while the
    posterior is right (review 2026-11-07: 1.08-1.14 with posterior sd within
    0.95-1.07 of a 4x longer run). ``bayesian_ci`` reports it as ``rhat_max``
    (``rhat_method`` "split_walkers") and judges the chain by its integrated
    autocorrelation time instead (ESS = N / tau, short under 50 tau).
    """
    c, n, p = chains.shape
    if n < 4:
        return np.inf
    half = n // 2
    X = chains[:, :2*half, :]
    X = X.reshape(c*2, half, p)
    W = X.var(axis=1, ddof=1).mean(axis=0)
    m = X.mean(axis=1)
    B = half * m.var(axis=0, ddof=1)
    var_hat = (half-1)/half * W + B/half
    rhat = np.sqrt(var_hat / W)
    return float(np.nanmax(rhat))
