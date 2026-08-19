import jax.numpy as jnp
import numpy as np


def is_uniform_grid(x, rtol=1e-5, atol=1e-8):
    """True if the 1D knot grid x has uniform spacing (checked at trace time)."""
    d = np.diff(np.asarray(x))
    return bool(np.allclose(d, d[0], rtol=rtol, atol=atol))


def cubic_knot_derivs(x, f):
    """Finite-difference knot derivatives along axis 0, identical to
    interpax approx_df(method="cubic") (_cubic1)."""
    dx = jnp.diff(x).reshape((-1,) + (1,) * (f.ndim - 1))
    df = jnp.diff(f, axis=0) / dx
    return jnp.concatenate([df[:1], 0.5 * (df[:-1] + df[1:]), df[-1:]], axis=0)


def uniform_cubic_interp1d(xq, x, f, fx=None, extrap=0.0):
    """Cubic interpolation on a UNIFORM knot grid.

    Same scheme as interpax.interp1d(method="cubic") (local cubic Hermite with
    finite-difference knot derivatives) but the bracketing index is computed in
    closed form instead of a per-query binary search, so it fuses under
    jit/vmap. Only valid for uniformly spaced x; check with is_uniform_grid.

    Args:
        xq: query points, any shape
        x: (n,) uniformly spaced knots
        f: (n, ...) values at knots
        fx: optional precomputed knot derivatives from cubic_knot_derivs
        extrap: fill value outside [x[0], x[-1]], or True to extrapolate

    Returns:
        interpolated values, shape xq.shape + f.shape[1:]
    """
    if fx is None:
        fx = cubic_knot_derivs(x, f)
    n = x.shape[0]
    step = (x[-1] - x[0]) / (n - 1)
    xq = jnp.asarray(xq)
    outshape = xq.shape + f.shape[1:]
    xq = jnp.atleast_1d(xq).reshape(-1)
    # matches clip(searchsorted(x, xq, side="right"), 1, n-1) on a uniform grid
    i = jnp.clip(jnp.floor((xq - x[0]) / step).astype(jnp.int32) + 1, 1, n - 1)
    x0 = x[i - 1]
    dx = x[i] - x0
    t = ((xq - x0) / dx).reshape((-1,) + (1,) * (f.ndim - 1))
    dxb = dx.reshape(t.shape)
    f0 = jnp.take(f, i - 1, axis=0)
    f1 = jnp.take(f, i, axis=0)
    fx0 = jnp.take(fx, i - 1, axis=0) * dxb
    fx1 = jnp.take(fx, i, axis=0) * dxb
    # Hermite coefficients on [0,1] (rows of interpax A_CUBIC)
    c2 = -3 * f0 + 3 * f1 - 2 * fx0 - fx1
    c3 = 2 * f0 - 2 * f1 + fx0 + fx1
    fq = f0 + t * (fx0 + t * (c2 + t * c3))
    if extrap is not True:
        out = ((xq < x[0]) | (xq > x[-1])).reshape(t.shape)
        fq = jnp.where(out, extrap, fq)
    return fq.reshape(outshape)


def Wvec(x, Delta=1, kind="linear"):
    """
    Compute basis function weights for spline interpolation.
    
    Args:
        x: Input array or scalar value
        Delta: Spacing between spline nodes (default: 1)
        kind: Type of interpolation, currently only "linear" supported (default: "linear")
    
    Returns:
        Basis function weights for linear spline interpolation
    """
    y = x / Delta

    if kind == "linear":
        return (y > -1) * (y <= 0) * (y + 1) + (y > 0) * (y <= 1) * (-y + 1)

    else:
        return 0


def spline_func_vec(xs, coeffs, xmin, Delta=1, kind="linear"):
    """
    Evaluate spline function at given points using coefficients.
    
    Args:
        xs: Array of x values where to evaluate the spline
        coeffs: List of coefficient arrays for each spline basis function
        xmin: Minimum x value of the spline domain
        Delta: Spacing between spline nodes (default: 1)
        kind: Type of interpolation, currently only "linear" supported (default: "linear")
    
    Returns:
        Array of spline function values evaluated at xs
    """
    ret = 0

    for ii, coeff in enumerate(coeffs):
        ret += coeff[:, None] * Wvec(xs - xmin - ii * Delta, Delta=Delta, kind=kind)

    return ret
