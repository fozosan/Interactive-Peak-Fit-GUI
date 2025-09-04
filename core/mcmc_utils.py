from __future__ import annotations
import numpy as np

def rhat_split(chains: np.ndarray) -> float:
    # chains: shape (n_chain, n_draw, n_param)
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

def ess_autocorr(chains: np.ndarray) -> float:
    # min ESS across params; Goodman–Weare chains concatenated
    c, n, p = chains.shape
    if n < 5:
        return float(min(c*n, 1))
    ess_min = float('inf')
    for j in range(p):
        x = chains[:, :, j].reshape(c*n)
        x = x - x.mean()
        acf = np.correlate(x, x, mode="full")[x.size-1:]
        acf = acf / acf[0]
        # Geyer initial positive sequence
        tau = 1.0
        for k in range(1, min(1000, acf.size)):
            if acf[k] + acf[k+1 if k+1 < acf.size else k] < 0:
                break
            tau += 2.0 * acf[k]
        ess = (c*n) / max(tau, 1.0)
        ess_min = min(ess_min, ess)
    return float(ess_min)
