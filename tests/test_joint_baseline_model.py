"""The joint baseline's columns and its linear solve (game plan 1.6, decision D2).

The baseline is a Legendre series over the fit window, in u = 2 (x - lo) /
(hi - lo) - 1, so its columns are centred and scaled whatever the x units: the
prior-art rule ``fit-affines-on-centered-coordinates`` (a solve on raw
coordinates is ill-conditioned and fails on thin data). Its coefficients are
free linear parameters, solved together with the peak heights, which stay
non-negative.
"""

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from core import baseline_model as BM


def test_core_default_is_fixed_and_batch_names_joint():
    assert BM.baseline_fit_choice({}) == "fixed"
    assert BM.baseline_fit_choice({"baseline_fit": None}) == "fixed"
    assert BM.baseline_fit_choice({"baseline_fit": "Joint"}) == "joint"
    assert BM.baseline_fit_choice({"baseline_fit": "fixed"}) == "fixed"
    with pytest.raises(ValueError, match="baseline_fit"):
        BM.baseline_fit_choice({"baseline_fit": "spline"})


def test_order_defaults_to_cubic_and_must_be_a_small_count():
    assert BM.DEFAULT_ORDER == 3
    assert BM.baseline_order({}) == 3
    assert BM.baseline_order({"baseline_order": 1}) == 1
    for bad in (-1, 2.5, "two"):
        with pytest.raises(ValueError, match="baseline_order"):
            BM.baseline_order({"baseline_order": bad})


def test_columns_are_legendre_in_the_centred_scaled_window():
    x = np.linspace(340.2, 439.7, 135)
    jb = BM.JointBaseline.over(x, 3)
    assert (jb.lo, jb.hi, jb.order, jb.n_coef) == (340.2, 439.7, 3, 4)
    B = jb.basis(x)
    u = 2.0 * (x - 340.2) / (439.7 - 340.2) - 1.0
    expect = np.column_stack([np.ones_like(u), u, 0.5 * (3 * u**2 - 1), 0.5 * (5 * u**3 - 3 * u)])
    np.testing.assert_allclose(B, expect, atol=1e-12)
    # well conditioned on its own window, as Legendre columns are on a uniform grid
    assert np.linalg.cond(B) < 5.0
    # outside the window the baseline stays at its edge values
    out = jb.basis(np.array([300.0, 500.0]))
    np.testing.assert_allclose(out, jb.basis(np.array([340.2, 439.7])))
    assert jb.names() == ["baseline_c0", "baseline_c1", "baseline_c2", "baseline_c3"]
    np.testing.assert_allclose(jb.evaluate(x, [2.0, 1.0, 0.0, 0.0]), 2.0 + u)


def test_columns_do_not_depend_on_where_the_window_sits():
    # The same window shifted by 1e4 cm-1 gives the same columns: the solve sees
    # centred, scaled coordinates, never raw ones.
    x = np.linspace(340.2, 439.7, 135)
    a = BM.JointBaseline.over(x, 3).basis(x)
    b = BM.JointBaseline.over(x + 1.0e4, 3).basis(x + 1.0e4)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_linear_solve_is_the_bounded_least_squares_optimum():
    rng = np.random.default_rng(5)
    n = 90
    x = np.linspace(-1.0, 1.0, n)
    A = np.column_stack([np.exp(-((x - c) / 0.1) ** 2) for c in (-0.3, 0.2, 0.6)])
    B = BM.JointBaseline.over(x, 2).basis(x)
    b = A @ [3.0, 0.0, 1.5] + B @ [10.0, -2.0, 0.7] + 0.3 * rng.standard_normal(n)
    w = 1.0 / (0.5 + rng.random(n))
    lin = BM.LinearPart(B * w[:, None])
    h, c = lin.solve(A * w[:, None], b * w)
    ref = lsq_linear(
        np.hstack([A, B]) * w[:, None],
        b * w,
        bounds=(np.r_[np.zeros(3), -np.inf * np.ones(3)], np.inf * np.ones(6)),
        tol=1e-14,
    )
    np.testing.assert_allclose(np.r_[h, c], ref.x, atol=1e-8)
    assert np.all(h >= 0.0)
    # a height that would go negative is held at zero, and the baseline takes up the rest
    b2 = A @ [3.0, 0.0, 1.5] - 0.8 * A[:, 1] + B @ [10.0, -2.0, 0.7]
    h2, c2 = lin.solve(A * w[:, None], b2 * w)
    ref2 = lsq_linear(
        np.hstack([A, B]) * w[:, None],
        b2 * w,
        bounds=(np.r_[np.zeros(3), -np.inf * np.ones(3)], np.inf * np.ones(6)),
        tol=1e-14,
    )
    assert h2[1] == 0.0
    np.testing.assert_allclose(np.r_[h2, c2], ref2.x, atol=1e-8)


def test_without_columns_the_solve_is_plain_nnls():
    rng = np.random.default_rng(6)
    A = rng.random((40, 2))
    b = A @ [1.0, 2.0] + 0.01 * rng.standard_normal(40)
    lin = BM.LinearPart(None)
    h, c = lin.solve(A, b)
    from scipy.optimize import nnls

    np.testing.assert_allclose(h, nnls(A, b)[0])
    assert c.size == 0
    M = rng.random((40, 3))
    np.testing.assert_array_equal(lin.project(M), M)


def test_projection_removes_exactly_the_baseline_columns():
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 1.0, 50)
    B = BM.JointBaseline.over(x, 3).basis(x) * (1.0 + rng.random(50))[:, None]
    lin = BM.LinearPart(B)
    M = rng.random((50, 4))
    P = lin.project(M)
    np.testing.assert_allclose(B.T @ P, 0.0, atol=1e-10)
    np.testing.assert_allclose(lin.project(B), 0.0, atol=1e-10)
