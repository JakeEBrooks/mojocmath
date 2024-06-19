from testing import assert_almost_equal, assert_equal, assert_true

from cmath import *

alias PI = 3.1415926535897932384626433
alias E = 2.7182818284590452353602874

fn complex_assert_almost_equal[
    type: DType, size: Int
](lhs: ComplexSIMD[type, size], 
    rhs: ComplexSIMD[type, size],
    /,
    *,
    msg: String = "",
    atol: Scalar[type] = 1e-06,
    rtol: Scalar[type] = 1e-05,
    equal_nan: Bool = False
) raises:
    assert_almost_equal(lhs.real, rhs.real, msg = msg, atol = atol, rtol = rtol, equal_nan = equal_nan)
    assert_almost_equal(lhs.imag, rhs.imag, msg = msg, atol = atol, rtol = rtol, equal_nan = equal_nan)

fn test_ispositive() raises:
    assert_equal(ispositive(SIMD[DType.int32, 2](-1, 1)), SIMD[DType.bool, 2](False, True))
    assert_equal(ispositive(SIMD[DType.float32, 2](-0., 0.)), SIMD[DType.bool, 2](False, True))

fn test_isnegative() raises:
    assert_equal(isnegative(SIMD[DType.int32, 2](-1, 1)), SIMD[DType.bool, 2](True, False))
    assert_equal(isnegative(SIMD[DType.float32, 2](-0., 0.)), SIMD[DType.bool, 2](True, False))

fn test_csqrt() raises:
    assert_true(csqrt(ComplexFloat32(-4., -0.)) == ComplexFloat32(+0., -2.))
    assert_true(csqrt(ComplexFloat32(-4., +0.)) == ComplexFloat32(+0., +2.))
    assert_true(csqrt(ComplexFloat32(+4., +0.)) == ComplexFloat32(+2., +0.))
    assert_true((csqrt(ComplexSIMD(SIMD[DType.float32, 2](-4., -4.), SIMD[DType.float32, 2](-0., +0.))) == 
                       ComplexSIMD(SIMD[DType.float32, 2](+0., +0.), SIMD[DType.float32, 2](-2., +2.))).reduce_and())

fn test_cis() raises:
    complex_assert_almost_equal(cis[DType.float32, 1](PI), ComplexFloat32(-1, 0), atol=1e-05)

fn test_clog() raises:
    complex_assert_almost_equal(clog(ComplexFloat32(10, 5)), ComplexFloat32(2.4141568686511508, 0.4636476090008061))
    complex_assert_almost_equal(clog(ComplexFloat32(E, 0)), ComplexFloat32(1, 0))

fn test_clog10() raises:
    complex_assert_almost_equal(clog10(ComplexFloat32(10, 0)), ComplexFloat32(1, 0))

fn test_clog2() raises:
    complex_assert_almost_equal(clog2(ComplexFloat32(2, 0)), ComplexFloat32(1, 0))

fn test_clogb() raises:
    complex_assert_almost_equal(clogb(ComplexFloat32(E, 0), E), clog(ComplexFloat32(E, 0)))

fn test_cexp() raises:
    complex_assert_almost_equal(cexp(ComplexFloat32(1, 0)), ComplexFloat32(E, 0))

fn test_cexp2() raises:
    complex_assert_almost_equal(cexp2(ComplexFloat32(4, 0)), ComplexFloat32(16, 0))

fn test_cpowc() raises:
    complex_assert_almost_equal(cpowc(ComplexFloat32(2, 0), ComplexFloat32(4, 0)), ComplexFloat32(16, 0))

fn test_powc() raises:
    complex_assert_almost_equal(powc(2, ComplexFloat32(4, 0)), ComplexFloat32(16, 0))

fn test_cpowi() raises:
    complex_assert_almost_equal(cpowi(ComplexFloat32(2, 0), Int32(4)), ComplexFloat32(16, 0))

fn test_cpowf() raises:
    complex_assert_almost_equal(cpowf(ComplexFloat32(2.5, 0), Float32(2.5)), ComplexFloat32(9.882117688026186, 0))

fn test_csquare() raises:
    complex_assert_almost_equal(csquare(ComplexFloat32(4, 0)), ComplexFloat32(16, 0))

fn test_csin() raises:
    complex_assert_almost_equal(csin(ComplexFloat32(PI/2, 0)), ComplexFloat32(1, -0.))

fn test_ccos() raises:
    complex_assert_almost_equal(ccos(ComplexFloat32(PI/2, 0)), ComplexFloat32(0, 0))

fn test_ctan() raises:
    complex_assert_almost_equal(ctan(ComplexFloat32(PI, 0)), ComplexFloat32(0, 0))

fn test_csinh() raises:
    complex_assert_almost_equal(csinh(ComplexFloat32(PI, 0)), ComplexFloat32(11.548739357257748, 0))

fn test_ccosh() raises:
    complex_assert_almost_equal(ccosh(ComplexFloat32(PI, 0)), ComplexFloat32(11.591953275521519, 0))

fn test_ctanh() raises:
    complex_assert_almost_equal(ctanh(ComplexFloat32(PI, 0)), ComplexFloat32(0.9962720762207496, 0))

fn test_casin() raises:
    complex_assert_almost_equal(casin(ComplexFloat32(PI, 0)), ComplexFloat32(1.5707963267948966, 1.8115262724608532))

fn test_cacos() raises:
    complex_assert_almost_equal(cacos(ComplexFloat32(PI, 0)), ComplexFloat32(0, -1.8115262724608532))

fn test_catan() raises:
    complex_assert_almost_equal(catan(ComplexFloat32(PI, 0)), ComplexFloat32(1.2626272556789118, 0))

fn test_casinh() raises:
    complex_assert_almost_equal(casinh(ComplexFloat32(PI, 0)), ComplexFloat32(1.8622957433108482, 0))

fn test_cacosh() raises:
    complex_assert_almost_equal(cacosh(ComplexFloat32(PI, 0)), ComplexFloat32(1.8115262724608532, 0))

fn test_catanh() raises:
    complex_assert_almost_equal(catanh(ComplexFloat32(PI, 0)), ComplexFloat32(0.32976531495669914, 1.5707963267948966))
    
fn main() raises:
    test_ispositive()
    test_isnegative()
    test_csqrt()
    test_cis()
    test_clog()
    test_clog10()
    test_clog2()
    test_clogb()
    test_cexp()
    test_cexp2()
    test_cpowc()
    test_powc()
    test_cpowi()
    test_cpowf()
    test_csquare()
    test_csin()
    test_ccos()
    test_ctan()
    test_csinh()
    test_ccosh()
    test_ctanh()
    test_casin()
    test_cacos()
    test_catan()
    test_casinh()
    test_cacosh()
    test_catanh()