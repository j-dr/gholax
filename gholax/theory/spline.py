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
