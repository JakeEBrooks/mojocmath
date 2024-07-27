from math import (sqrt, copysign, sin, sinh,
                    cos, cosh, log, exp)

from cmath.complexsimd import ComplexSIMD

# A note on notation...
# In comments throughout, an example complex value z = a + bi is used.
# z has absolute value r, and argument x
# The result of an operation is often a complex value w = c + di

# Sources used:
# The Rust stdlib implementations
# Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign Bit, by Kahan, W.
# Handbook of Mathematical Functions: With Formulas, Graphs, and Mathematical Tables, by Milton Abramowitz, Irene A. Stegun
# ... and Wikipedia :)

alias _R2 = 1.414213562373095048801688724209698 # Square root of 2

# General Functions

fn ispositive[type: DType, width: Int](val: SIMD[type, width]) -> SIMD[DType.bool, width]:
    """Returns a `SIMD` where the element at index i is `True` if `val[i]` is positive.

    Parameters:
        type: The input datatype.
        width: The input `SIMD` width.

    Args:
        val: The `SIMD` to evaluate.
    """
    return copysign(1, val) == 1

fn isnegative[type: DType, width: Int](val: SIMD[type, width]) -> SIMD[DType.bool, width]:
    """Returns a `SIMD` where the element at index i is `True` if `val[i]` is negative.

    Parameters:
        type: The input datatype.
        width: The input `SIMD` width.

    Args:
        val: The `SIMD` to evaluate.
    """
    return copysign(1, val) == -1

fn csqrt[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise square root operation on the complex values in `z`. Each output value, w, is the principal
    square root such that `arg(w)` lies in the range [-π, π].

    This function has a branch cut along the negative real axis: (-∞, 0)

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.
    
    Returns:
        A `ComplexSIMD` containing the elementwise principal square roots of the complex values in `z`. 
    """

    # For a complex number z = a + bi, the principal square root of z is given by:
    #   sqrt(z) = sqrt( (H + a)/2 ) + sign(b)*sqrt( (H - a)/2 )
    # where H is the absolute value of z, |z|
    # 
    # But this formula can be simplified when either a or b are zero, and there are conventions
    # on the behaviour of this function along the negative real axis. The following checks are implemented
    # in this function for a given complex number z = a + bi:
    #
    # If b = 0 and a is positive, then it is a simple square root e.g.
    #   sqrt(4 + 0i) = 2 + 0i
    # If b = 0 and a is negative, then it is a pure complex number where the sign of the 
    #   imaginary part is carried through e.g. 
    #   sqrt(-4 + 0i) = 0 + 2i
    #   sqrt(-4 - 0i) = 0 - 2i
    # If a = 0, then the above formula can be greatly simplified to:
    #   sqrt(z) = sqrt(abs(b)/2) + sign(b)*sqrt(abs(b)/2)
    # Otherwise, the above formula is used in full

    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()

    var u: SIMD[type, width] # The real part of the output
    var v: SIMD[type, width] # The imaginary part of the output
    var H: SIMD[type, width] # The absolute value of the input, |z|

    var rsign: SIMD[DType.bool, width] # The signs of the real values
    var r0s: SIMD[DType.bool, width] # If real values are 0
    var i0s: SIMD[DType.bool, width] # If imaginary values are 0

    if (z.real != 0).reduce_and() and (z.imag != 0).reduce_and():
        # All elements are non-zero, so we don't care about the branch cut
        H = z.abs()
        u = sqrt(H + z.real) / _R2
        v = copysign(sqrt(H - z.real) / _R2, z.imag)
        return ComplexSIMD(u, v)
    
    else:
        # Some elements are zero, so handle the branch cut
        H = z.abs()
        rsign = ispositive(z.real)
        r0s = z.real == 0
        i0s = z.imag == 0

        # This might be doing some unecessary ops (and is kinda ugly)
        u = i0s.select(rsign.select(sqrt(z.real), 0), r0s.select(abs(z.imag/_R2), sqrt(H + z.real)/_R2))
        v = i0s.select(rsign.select(0, copysign(sqrt(abs(z.real)), z.imag)), r0s.select(copysign(z.imag/_R2, z.imag), copysign(sqrt(H - z.real) / _R2, z.imag)))

        return ComplexSIMD(u, v)

fn cis[type: DType, width: Int](a: SIMD[type, width]) -> ComplexSIMD[type, width]:
    """Returns a `ComplexSIMD` where the element at index i is the result of `cos(a[i]) + isin(a[i])`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `SIMD` width.

    Args:
        a: The `SIMD` to evaluate.
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # cis(a) = cos(a) + isin(a)

    return ComplexSIMD(cos(a), sin(a))

# Logarithmic Functions

fn clog[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise natural logarithm operation on the complex values in `z`. This is a logarithm
    operation with base e, where e is Euler's Number. Each output value, w = c + di, is the principal value 
    of the natural logarithm such that d falls in the range [-π, π].

    This functions contains a branch cut along the negative real axis: (-∞, 0]

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the elementwise natural logarithm of the complex values in `z`. 
    """

    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # z = r * e^(ix)
    # So log(z) = log(r) + ix

    var r = z.abs()
    var x = z.arg()
    return ComplexSIMD(log(r), x)

fn clog10[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise base-10 logarithm operation on the complex values within `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the elementwise base-10 logarithm of the complex values in `z`.
    """

    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # Given an arbitrary base b:
    # log_b(z) = log(z)/log(b)

    return clog(z)/log(SIMD[type, width](10))

fn clog2[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise base-2 logarithm operation on the complex values within `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the elementwise base-2 logarithm of the complex values in `z`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # Given an arbitrary base b:
    # log_b(z) = log(z)/log(b)

    return clog(z)/log(SIMD[type, width](2))

fn clogb[type: DType, width: Int](z: ComplexSIMD[type, width], b: SIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise logarithm operation on the complex values within `z` with specified bases `b`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.
        b: The logarithm bases to use in evaluating `z`.

    Returns:
        A `ComplexSIMD` containing the elementwise base-b logarithm of the complex values in `z`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # Given an arbitrary base b:
    # log_b(z) = log(z)/log(b)

    return clog(z)/log(b)

# Exponential Functions

fn cexp[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise exponential operation, `e^z`, where e is Euler's Number.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing e^z. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # z = a + bi, so e^z = e^(a + bi) = e^a * e^(bi)
    # From Euler's formula, e^(bi) = cos(b) + isin(b)

    var e_a = exp(z.real)
    return ComplexSIMD(e_a*cos(z.imag), e_a*sin(z.imag))

fn cexp2[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise exponential operation, `2^z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `2^z`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # If w = 2^z, then log(w) = z*log(2), so w = e^(z*log(2))

    return cexp(z*log(SIMD[type, width](2)))

fn cpowc[type: DType, width: Int](z: ComplexSIMD[type, width], n: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise exponential operation, `z^n`, where the base `z` and the exponent
    `n` are both complex.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to use as a base.
        n: The `ComplexSIMD` to use as an exponent.

    Returns:
        A `ComplexSIMD` containing the result of computing `z^n`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # If w = z^n, then log(w) = n*log(z), so w = e^(n*log(z))

    return cexp(n * clog(z))

fn powc[type: DType, width: Int](a: SIMD[type, width], z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise exponential operation, `a^n`, where the base `a` is real and the exponent
    `n` is complex.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `SIMD` width.

    Args:
        a: The real `SIMD` to use as a base.
        z: The `ComplexSIMD` to use as an exponent.

    Returns:
        A `ComplexSIMD` containing the result of computing `a^z`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # If w = a^z, then log(w) = z*log(a), so w = e^(z*log(a))

    return cexp(z * log(a))

fn cpowi[type: DType, width: Int, ntype: DType](z: ComplexSIMD[type, width], n: SIMD[ntype, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise complex exponential operation, `z^n`, where the exponent `n` is a `SIMD` of integers.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `SIMD` width.
        ntype: The exponent datatype. Must be integral.

    Args:
        z: The `ComplexSIMD` to evaluate.
        n: The `SIMD` of integers to use as an exponent.

    Returns:
        A `ComplexSIMD` containing the result of computing `z^n`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    constrained[ntype.is_integral(), "The exponent datatype must be integral."]()
    # Using De Moivre's Formula:
    # z^n = r^n * (cos(nx) + isin(nx))

    var r = z.abs()
    var x = z.arg()
    var n_flt = n.cast[type]()
    var rpown = r**n_flt

    return ComplexSIMD(rpown*cos(n_flt*x), rpown*sin(n_flt*x))

fn cpowf[type: DType, width: Int, ntype: DType](z: ComplexSIMD[type, width], n: SIMD[ntype, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise complex exponential operation, `z^n`, where the exponent `n` is a `SIMD` of
    floating-point values.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `SIMD` width.
        ntype: The exponent datatype. Must be floating-point.

    Args:
        z: The `ComplexSIMD` to evaluate.
        n: The `SIMD` of floats to use as an exponent.

    Returns:
        A `ComplexSIMD` containing the result of computing `z^n`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    constrained[ntype.is_floating_point(), "The exponent datatype must be a float."]()
    # z^n = (r * e^(ix))^n = r^n * e^(ixn)
    # Since for this number, arg = xn, z^n = from_polar(r^n, xn)

    var r = z.abs()
    var x = z.arg()
    var n_flt = n.cast[type]()
    return ComplexSIMD.from_polar(r**n_flt, x*n_flt)

fn cpow[type: DType, width: Int, ntype: DType](z: ComplexSIMD[type, width], n: SIMD[ntype, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise complex exponential operation, `z^n`, where `n` can be any real value.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `SIMD` width.
        ntype: The exponent datatype.

    Args:
        z: The `ComplexSIMD` to evaluate.
        n: The `SIMD` to use as an exponent.

    Returns:
        A `ComplexSIMD` containing the result of computing `z^n`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # Resolves to either cpowi or cpowf based on datatype
    @parameter
    if ntype.is_integral():
        return cpowi(z, n)
    else:
        return cpowf(z, n)

fn csquare[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """
    Compute the common case of `z ^ 2`. This does not require a floating-point input.

    Parameters:
        type: The input datatype.
        width: The input `SIMD` width.

    Args:
        z: The base complex number.

    Returns:
        The value of `z ^ 2`.
    """
    # (a + bi)^2 = (a - b)(a + b) + 2abi
    var u = (z.real - z.imag)*(z.real + z.imag)
    var v = 2*z.real*z.imag
    return ComplexSIMD(u, v)

# Trigonometric Functions

fn csin[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise sine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `sin(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # sin(z) = sin(a)*cosh(b) + cos(a)*sinh(b)i

    return ComplexSIMD(sin(z.real)*cosh(z.imag), cos(z.real)*sinh(z.imag))

fn ccos[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise cosine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `cos(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # cos(z) = cos(a)*cosh(b) + sin(a)*sinh(b)i

    return ComplexSIMD(cos(z.real)*cosh(z.imag), sin(z.real)*sinh(z.imag))

fn ctan[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise tangent operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `tan(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # tan(z) = (sin(2a) + isinh(2b))/(cos(2a) + cosh(2b))

    var two_a = 2*z.real
    var two_b = 2*z.imag
    var denom = cos(two_a) + cosh(two_b)

    return ComplexSIMD(sin(two_a)/denom, sinh(two_b)/denom)

# Hyperbolic Functions

fn csinh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise hyperbolic sine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `sinh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # sinh(z) = sinh(a)*cos(b) + icosh(a)*sin(b)

    return ComplexSIMD(sinh(z.real)*cos(z.imag), cosh(z.real)*sin(z.imag))

fn ccosh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise hyperbolic cosine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `cosh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # cosh(z) = cosh(a)*cos(b) + isinh(a)*sin(b)

    return ComplexSIMD(cosh(z.real)*cos(z.imag), sinh(z.real)*sin(z.imag))

fn ctanh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise hyperbolic tangent operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `tanh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # tanh(z) = (sinh(2a) + isin(2b))/(cosh(2a) + cos(2b))

    var two_a = 2*z.real
    var two_b = 2*z.imag
    var denom = cosh(two_a) + cos(two_b)

    return ComplexSIMD(sinh(two_a)/denom, sin(two_b)/denom)

# Inverse Trigonometric Functions

fn casin[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse sine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `asin(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # asin(z) = -i*ln(sqrt(1 - z^2) + iz)

    var i = ComplexSIMD[type, width](SIMD[type, width](0), SIMD[type, width](1))

    return -i*clog(csqrt(1 - csquare(z)) + i*z)

fn cacos[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse cosine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `acos(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # acos(z) = -i*ln(i*sqrt(1 - z^2) + z)

    var i = ComplexSIMD[type, width](SIMD[type, width](0), SIMD[type, width](1))

    return -i*clog(i*csqrt(1 - csquare(z)) + z)

fn catan[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse tangent operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `atan(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # atan(z) = -(i/2)*ln((i - z)/(i + z))

    var i = ComplexSIMD[type, width](SIMD[type, width](0), SIMD[type, width](1))
    var i2 = ComplexSIMD[type, width](SIMD[type, width](0), SIMD[type, width](0.5))

    return -i2*clog((i - z)/(i + z))

# Inverse Hyperbolic Functions

fn casinh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse hyperbolic sine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `asinh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # asinh(z) = log(z + sqrt(z^2 + 1))

    return clog(z + csqrt(csquare(z) + 1))

fn cacosh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse hyperbolic cosine operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `acosh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # acosh(z) = log(z + sqrt(z + 1)*sqrt(z - 1))

    return clog(z + csqrt(z + 1)*csqrt(z - 1))

fn catanh[type: DType, width: Int](z: ComplexSIMD[type, width]) -> ComplexSIMD[type, width]:
    """Performs an elementwise inverse hyperbolic tangent operation on the complex values in `z`.

    Parameters:
        type: The input datatype. Must be a floating-point datatype.
        width: The input `ComplexSIMD` width.

    Args:
        z: The `ComplexSIMD` to evaluate.

    Returns:
        A `ComplexSIMD` containing the result of computing `atanh(z)`. 
    """
    constrained[type.is_floating_point(), "The input datatype must be a floating-point datatype."]()
    # atanh(z) = (1/2)*log((1 + z)/(1 - z))

    return 0.5*clog((1 + z)/(1 - z))