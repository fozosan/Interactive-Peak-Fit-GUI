"""The ALS baseline solves its banded system directly.

``core.signals.als_baseline`` used to build dense L x L matrices and solve them
with ``np.linalg.solve``: 2.8-4.4 s for one 1650-point spectrum on a quiet
machine, and opening a file froze the GUI for 45 s under load. The system is
pentadiagonal, so a banded solve gives the same baseline in milliseconds. These
tests hold the new solve to the old dense result.
"""

from contextlib import nullcontext
import time

import numpy as np
import pytest

from core import signals

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None


def _dense_als(y, lam=1e5, p=0.001, niter=10, tol=0.0):
    """``core.signals.als_baseline`` as it was before the banded solve, verbatim."""
    y = np.asarray(y, dtype=float)
    L = y.size
    # second-order difference matrix; diff along rows to keep square result
    D = np.diff(np.eye(L), 2, axis=0)
    w = np.ones(L)
    z = np.zeros_like(y)
    for _ in range(int(niter)):
        W = np.diag(w)
        Z = np.linalg.solve(W + lam * D.T @ D, w * y)
        if tol and np.sqrt(np.mean((Z - z) ** 2)) < tol:
            z = Z
            break
        z = Z
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def _dense_reference(y, **kwargs):
    # One BLAS thread keeps the dense reference near 2 s per 1650 points; the
    # default pool takes 3-20 s depending on load.
    limit = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
    with limit:
        return _dense_als(y, **kwargs)


def _raman_spectrum(kind, seed=0):
    """A MoS2 Raman spectrum like the 800g-grating files: 1650 points, -133 to 1069 cm-1.

    A dark offset near 504 counts, peaks a few cm-1 wide, and count noise with
    variance 0.724 + 0.0451 * signal, the model measured on those spectra.
    ``"mos2"`` has a flat background and the E2g and A1g lines; ``"fluorescence"``
    adds a broad sloped background and a strong silicon line at 520 cm-1.
    """
    x = np.linspace(-133.0, 1069.0, 1650)
    rng = np.random.default_rng(seed)
    clean = np.full_like(x, 504.0)
    lines = [(383.0, 65.0, 4.0), (408.0, 90.0, 5.0)]
    if kind == "fluorescence":
        clean = clean + 40.0 * np.exp(-(((x - 300.0) / 450.0) ** 2)) + 0.004 * x
        lines = lines + [(520.0, 600.0, 3.5)]
    for center, height, fwhm in lines:
        clean = clean + height / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)
    signal = np.clip(clean - 504.0, 0.0, None)
    y = clean + rng.normal(0.0, np.sqrt(0.724 + 0.0451 * signal))
    return x, y


@pytest.mark.parametrize(
    "kind, window, params",
    [
        # The GUI defaults on a whole spectrum.
        ("mos2", None, dict(lam=1e5, p=0.001, niter=10, tol=0.0)),
        # tol stops the loop at iteration 7 of 10. Stopping one iteration early
        # or late moves the baseline by 4.9e-4 or 9.3e-5 relative, far outside
        # the tolerance.
        ("fluorescence", None, dict(lam=1e5, p=0.001, niter=10, tol=0.3)),
        # Fit windows (138 points), as the GUI and batch compute them.
        ("mos2", (340.0, 440.0), dict(lam=1e5, p=0.01, niter=10, tol=0.0)),
        ("fluorescence", (340.0, 440.0), dict(lam=1e7, p=0.001, niter=10, tol=0.0)),
        ("mos2", (340.0, 440.0), dict(lam=1e3, p=0.05, niter=3, tol=0.0)),
    ],
)
def test_matches_dense_solve_on_raman_spectra(kind, window, params):
    x, y = _raman_spectrum(kind)
    if window is not None:
        y = y[(x >= window[0]) & (x <= window[1])]

    z = signals.als_baseline(y, **params)

    np.testing.assert_allclose(z, _dense_reference(y, **params), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("niter", [0, 1, 10])
def test_short_inputs_match_dense_solve(n, niter):
    # The ends of the band, where the difference matrix is truncated, and the
    # degenerate lengths below 3, where it has no rows at all.
    y = 500.0 + 3.0 * np.sin(np.arange(n, dtype=float)) + np.arange(n, dtype=float) ** 1.5

    z = signals.als_baseline(y, lam=10.0, p=0.1, niter=niter)

    np.testing.assert_allclose(z, _dense_als(y, lam=10.0, p=0.1, niter=niter), rtol=1e-9, atol=1e-9)


def test_full_spectrum_baseline_takes_milliseconds():
    _, y = _raman_spectrum("mos2")
    signals.als_baseline(y)  # first call pays for imports

    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        signals.als_baseline(y, lam=1e5, p=0.001, niter=10, tol=0.0)
        best = min(best, time.perf_counter() - t0)

    assert best < 0.1, f"1650-point ALS baseline took {best:.3f} s"
