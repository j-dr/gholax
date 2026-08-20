"""Uniform-grid cubic interpolator vs interpax.interp1d(method="cubic").

The uniform-grid path replaces interpax's per-query binary search with
closed-form indexing but must reproduce the same local cubic Hermite scheme
to fp32 tolerance, in values and gradients, interior and edges.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interpax import interp1d

sys.path.insert(0, ".")
from gholax.theory.spline import (  # noqa: E402
    cubic_knot_derivs,
    is_uniform_grid,
    uniform_cubic_interp1d,
)

RTOL = 1e-5  # fp32: normalized against max |reference|
N = 100
X = jnp.linspace(-3.0, 1.0, N)


def _f_smooth():
    return jnp.sin(3.0 * X) + 0.5 * X**2


def _assert_close(a, b, rtol=RTOL):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), 1e-30)
    np.testing.assert_allclose(b / scale, a / scale, rtol=0, atol=rtol)


def test_is_uniform_grid():
    assert is_uniform_grid(np.linspace(0, 1, 50))
    assert not is_uniform_grid(np.geomspace(1, 10, 50))


@pytest.mark.parametrize("extrap", [0.0, True])
def test_values_interior(extrap):
    f = _f_smooth()
    xq = jnp.asarray(np.random.default_rng(1).uniform(-2.9, 0.9, 500), dtype=X.dtype)
    a = interp1d(xq, X, f, extrap=extrap, method="cubic")
    b = uniform_cubic_interp1d(xq, X, f, extrap=extrap)
    _assert_close(a, b)


@pytest.mark.parametrize("extrap", [0.0, True])
def test_values_edges_and_outside(extrap):
    f = _f_smooth()
    # knots at the boundary, boundary values, and slightly out-of-range points
    xq = jnp.concatenate(
        [X[:3], X[-3:], jnp.array([-3.0, 1.0, -3.0000005, 1.0000005])]
    )
    a = interp1d(xq, X, f, extrap=extrap, method="cubic")
    b = uniform_cubic_interp1d(xq, X, f, extrap=extrap)
    _assert_close(a, b)
    if extrap is not True:
        # matches interpax _extrap: strictly-outside points get the fill value
        assert np.all(np.asarray(b)[-2:] == extrap)


def test_far_extrapolation():
    # Far outside the knots the end-cell cubic is evaluated at |t| >> 1, which
    # amplifies fp32 rounding by ~|t|^3 in both implementations; agreement is
    # correspondingly looser there (indices and scheme still identical).
    f = _f_smooth()
    xq = jnp.array([-3.5, 1.5])
    a = interp1d(xq, X, f, extrap=True, method="cubic")
    b = uniform_cubic_interp1d(xq, X, f, extrap=True)
    _assert_close(a, b, rtol=1e-3)


def test_values_2d_f():
    f = jnp.stack([_f_smooth(), jnp.cos(2.0 * X)], axis=1)
    xq = jnp.asarray(np.random.default_rng(2).uniform(-3.1, 1.1, 300), dtype=X.dtype)
    a = interp1d(xq, X, f, extrap=0.0, method="cubic")
    b = uniform_cubic_interp1d(xq, X, f, extrap=0.0)
    assert a.shape == b.shape
    _assert_close(a, b)


def test_values_at_knots():
    f = _f_smooth()
    a = interp1d(X, X, f, extrap=0.0, method="cubic")
    b = uniform_cubic_interp1d(X, X, f, extrap=0.0)
    _assert_close(a, b)


def test_scalar_query():
    f = _f_smooth()
    a = interp1d(jnp.asarray(0.3, X.dtype), X, f, extrap=0.0, method="cubic")
    b = uniform_cubic_interp1d(jnp.asarray(0.3, X.dtype), X, f, extrap=0.0)
    assert np.shape(a) == np.shape(b)
    _assert_close(a, b)


@pytest.mark.parametrize("extrap", [0.0, True])
def test_gradients(extrap):
    f = _f_smooth()
    xq = jnp.concatenate(
        [
            jnp.asarray(
                np.random.default_rng(3).uniform(-2.9, 0.9, 200), dtype=X.dtype
            ),
            X[:2],
            X[-2:],
        ]
    )

    ga = jax.grad(
        lambda ff: jnp.sum(interp1d(xq, X, ff, extrap=extrap, method="cubic"))
    )(f)
    gb = jax.grad(lambda ff: jnp.sum(uniform_cubic_interp1d(xq, X, ff, extrap=extrap)))(
        f
    )
    _assert_close(ga, gb)

    ha = jax.grad(
        lambda q: jnp.sum(interp1d(q, X, f, extrap=extrap, method="cubic"))
    )(xq)
    hb = jax.grad(lambda q: jnp.sum(uniform_cubic_interp1d(q, X, f, extrap=extrap)))(xq)
    _assert_close(ha, hb, rtol=1e-4)


def test_precomputed_fx():
    f = _f_smooth()
    xq = jnp.asarray(np.random.default_rng(4).uniform(-2.9, 0.9, 100), dtype=X.dtype)
    fx = cubic_knot_derivs(X, f)
    a = uniform_cubic_interp1d(xq, X, f, extrap=0.0)
    b = uniform_cubic_interp1d(xq, X, f, fx=fx, extrap=0.0)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
