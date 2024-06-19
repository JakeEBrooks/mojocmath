from math import atan2, hypot, cos, sin

from cmath.complexmath import cpow, cpowc, powc

alias ComplexScalar = ComplexSIMD[width = 1]
"""Represents a complex scalar dtype."""

alias ComplexInt8 = ComplexScalar[DType.int8]
"""Represents a complex 8-bit signed integer."""
alias ComplexInt16 = ComplexScalar[DType.int16]
"""Represents a complex 16-bit signed integer."""
alias ComplexInt32 = ComplexScalar[DType.int32]
"""Represents a complex 32-bit signed integer."""
alias ComplexInt64 = ComplexScalar[DType.int64]
"""Represents a complex 64-bit signed integer."""

alias ComplexUInt8 = ComplexScalar[DType.uint8]
"""Represents a complex 8-bit unsigned integer."""
alias ComplexUInt16 = ComplexScalar[DType.uint16]
"""Represents a complex 16-bit unsigned integer."""
alias ComplexUInt32 = ComplexScalar[DType.uint32]
"""Represents a complex 32-bit unsigned integer."""
alias ComplexUInt64 = ComplexScalar[DType.uint64]
"""Represents a complex 64-bit unsigned integer."""

alias ComplexFloat16 = ComplexScalar[DType.float16]
"""Represents a complex 16-bit floating point value."""
alias ComplexFloat32 = ComplexScalar[DType.float32]
"""Represents a complex 32-bit floating point value."""
alias ComplexFloat64 = ComplexScalar[DType.float64]
"""Represents a complex 64-bit floating point value."""

@register_passable("trivial")
struct ComplexSIMD[type: DType, width: Int](Sized, CollectionElement, Stringable, Absable, Powable):
    """
    Represents complex values as a pair of `SIMD` vectors.

    Constraints:
        As with normal `SIMD` vectors, the size of a `ComplexSIMD` must also be positive and a power of 2.
    
    Parameters:
        type: The datatype of the `SIMD` elements.
        width: The size of the `SIMD` vectors.
    """
    alias _type = SIMD[type, width]

    var real: Self._type
    """The real part of the `SIMD` vector."""
    var imag: Self._type
    """The imaginary part of the `SIMD` vector."""

    fn __init__() -> Self:
        """Default initializer where all elements are set to zero.

        Returns:
            A `ComplexSIMD` where all elements are set to zero.
        """
        return Self{real: Self._type(),
                    imag: Self._type()}
    
    fn __init__(value: Self._type) -> Self:
        """Initialise a `ComplexSIMD` with a real component equal to `value` and an imaginary component equal to zero.

        Args:
            value: The `SIMD` used to initialise the real part.

        Returns:
            A `ComplexSIMD` with a real component equal to `value` and a zero imaginary component.
        """
        return Self{real: value, imag: Self._type()}
    
    fn __init__(real: Self._type, imag: Self._type) -> Self:
        """Initialise a `ComplexSIMD` from a pair of `SIMD` vectors representing real and imaginary components.

        Args:
            real: A `SIMD` representing the real part.
            imag: A `SIMD` representing the imaginary part.

        Returns:
            A `ComplexSIMD` initialised with the values of `real` and `imag`.
        """
        return Self{real: real, imag: imag}
    
    fn __add__(self, other: Self) -> Self:
        """Compute the complex addition of `self + other`.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `self + other`.
        """
        return Self(self.real+other.real, self.imag+other.imag)
    
    fn __add__(self, other: Self._type) -> Self:
        """Compute the real addition `self + other`.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self + other`.
        """
        return Self(self.real+other, self.imag)
    
    fn __radd__(self, other: Self) -> Self:
        """Compute the complex addition `other + self`.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `other + self`.
        """
        return self + other
    
    fn __radd__(self, other: Self._type) -> Self:
        """Compute the real addition `other + self`.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `other + self`.
        """
        return self + other
    
    fn __iadd__(inout self, other: Self):
        """Compute the in-place complex addition of `self` and `other`.

        Args:
            other: A `ComplexSIMD` of complex values.
        """
        self.real += other.real
        self.imag += other.imag
    
    fn __iadd__(inout self, other: Self._type):
        """Compute the in-place real addition of `self` and `other`.

        Args:
            other: A `SIMD` of real values.
        """
        self.real += other
    
    fn __sub__(self, other: Self) -> Self:
        """Compute the complex subtraction `self - other`.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `self - other`.
        """
        return Self(self.real-other.real, self.imag-other.imag)
    
    fn __sub__(self, other: Self._type) -> Self:
        """Compute the real subtraction `self - other`.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self - other`.
        """
        return Self(self.real-other, self.imag)
    
    fn __rsub__(self, other: Self) -> Self:
        """Compute the complex subtraction `other - self`.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `other - self`.
        """
        return -self + other
    
    fn __rsub__(self, other: Self._type) -> Self:
        """Compute the real subtraction `other - self`.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `other - self`.
        """
        return -self + other

    fn __isub__(inout self, other: Self):
        """Compute the in-place complex subtraction of `self` and `other`.

        Args:
            other: A `ComplexSIMD` of complex values.
        """
        self.real -= other.real
        self.imag -= other.imag
    
    fn __isub__(inout self, other: Self._type):
        """Compute the in-place real subtraction of `self` and `other`.

        Args:
            other: A `SIMD` of real values.
        """
        self.real -= other
    
    fn __mul__(self, other: Self) -> Self:
        """Compute the value of `self * other`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `self * other`.
        """
        return Self(self.real*other.real - self.imag*other.imag,
                    self.real*other.imag + self.imag*other.real)
    
    fn __mul__(self, other: Self._type) -> Self:
        """Compute the value of `self * other`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self * other`.
        """
        return Self(self.real*other, self.imag*other)
    
    fn __rmul__(self, other: Self) -> Self:
        """Compute the value of `other * self`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `other * self`.
        """
        return self * other

    fn __rmul__(self, other: Self._type) -> Self:
        """Compute the value of `other * self`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `other * self`.
        """
        return self * other
    
    fn __imul__(inout self, other: Self):
        """Compute the in-place multiplication of `self` and `other`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.
        """
        self.real = self.real*other.real - self.imag*other.imag
        self.imag = self.real*other.imag + self.imag*other.real
    
    fn __imul__(inout self, other: Self._type):
        """Compute the in-place multiplication of `self` and `other`, where `other` is real.

        Args:
            other: A `SIMD` of real values.
        """
        self.real = self.real*other
        self.imag = self.imag*other
    
    fn __truediv__(self, other: Self) -> Self:
        """Compute the value of `self / other`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `self / other`.
        """
        var denom = other.real*other.real + other.imag*other.imag
        return Self((self.real*other.real + self.imag*other.imag)/denom,
                    (self.imag*other.real - self.real*other.imag)/denom)
    
    fn __truediv__(self, other: Self._type) -> Self:
        """Compute the value of `self / other`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self / other`.
        """
        return Self(self.real/other, self.imag/other)
    
    fn __rtruediv__(self, other: Self) -> Self:
        """Compute the value of `other / self`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `other / self`.
        """
        var denom = self.real*self.real + self.imag*self.imag
        return Self((other.real*self.real + other.imag*self.imag)/denom,
                    (other.imag*self.real - other.real*self.imag)/denom)
    
    fn __rtruediv__(self, other: Self._type) -> Self:
        """Compute the value of `other / self`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `other / self`.
        """
        var denom = self.real*self.real + self.imag*self.imag
        return Self((other*self.real)/denom,
                    (-other*self.imag)/denom)
    
    fn __floordiv__(self, other: Self._type) -> Self:
        """Compute the value of `self // other`, where `other` is real. More specifically, the output
        real component is equal to `self.real // other` and the output imaginary component is equal
        to `self.imag // other`.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self // other`.
        """
        return Self(self.real//other, self.imag//other)
    
    fn __pow__(self, other: Self) -> Self:
        """Compute the value of `self ** other`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `self ** other`.
        """
        return cpowc(self, other)
    
    fn __pow__(self, other: Self._type) -> Self:
        """Compute the value of `self ** other`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `self ** other`.
        """
        return cpow(self, other)
    
    fn __rpow__(self, other: Self) -> Self:
        """Compute the value of `other ** self`, where `other` is complex.

        Args:
            other: A `ComplexSIMD` of complex values.

        Returns:
            A `ComplexSIMD` representing `other ** self`.
        """
        return cpowc(other, self)

    fn __rpow__(self, other: Self._type) -> Self:
        """Compute the value of `other ** self`, where `other` is real.

        Args:
            other: A `SIMD` of real values.

        Returns:
            A `ComplexSIMD` representing `other ** self`.
        """
        return powc(other, self)
    
    fn __neg__(self) -> Self:
        """Negates both compinents of the `ComplexSIMD`.

        Returns:
            A `ComplexSIMD` representing `- self`.
        """
        return Self(-self.real, -self.imag)

    fn __str__(self) -> String:
        """Get a string representation of `self`.
        
        Returns:
            A string representation of `self`.
        """
        return str(self.real)+" "+str(self.imag)+"i"
    
    fn __len__(self) -> Int:
        """Get the `SIMD` width of `self`.
        
        Returns:
            An `Int` representing the `SIMD` width of both components of `self`.
        """
        return width
    
    fn __getitem__(self, index: Int) -> ComplexScalar[type]:
        """Gets an element from the `ComplexSIMD` as a `ComplexScalar`.

        Args:
            index: The element index.

        Returns:
            The complex value at position `index`.
        """
        return ComplexScalar[type](self.real[index], self.imag[index])
    
    fn __setitem__(inout self, index: Int, value: ComplexScalar[type]):
        """Sets the element at `index` in `self` using the complex value in `value`.

        Args:
            index: The element index.
            value: The `ComplexScalar` to set from.
        """
        self.real[index] = value.real
        self.imag[index] = value.imag
    
    fn __eq__(self, other: Self) -> SIMD[DType.bool, width]:
        """Compares two `ComplexSIMD` vectors using equal-to comparison.

        Args:
            other: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True if the real and imaginary components of the inputs are equal at position `i`.
            In other words, the values `self.real[i] == other.real[i]` and `self.imag[i] == other.imag[i]` are both `True`.
        """
        return (self.real == other.real) & (self.imag == other.imag)

    fn __ne__(self, other: Self) -> SIMD[DType.bool, width]:
        """Compares two `ComplexSIMD` vectors using not equal-to comparison.

        Args:
            other: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True if the real and imaginary components of the inputs are not equal at position `i`.
            In other words, the values of `self.real[i] != other.real[i]` and `self.imag[i] != other.imag[i]` are both `True`.
        """
        return (self.real != other.real) | (self.imag != other.imag)
    
    @staticmethod
    fn splat(val: Scalar[type]) -> Self:
        """Create a new `ComplexSIMD` by splatting `val` across the real component.

        Args:
            val: The input scalar value.

        Returns:
            A new `ComplexSIMD` whose real elements are the same as the input value. The imaginary elements
            are all set to zero.
        """
        return Self{real: SIMD[type, width].splat(val),
                    imag: SIMD[type, width]()}

    @staticmethod
    fn splat(real: Scalar[type], imag: Scalar[type]) -> Self:
        """Create a new `ComplexSIMD` by splatting `real` across the real component and `imag` across the imaginary component.

        Args:
            real: The input scalar value for the real component.
            imag: The input scalar value for the imaginary component.

        Returns:
            A new `ComplexSIMD` whose real elements are all equal to `real` and whose imaginary elements are all equal to `imag`.
        """
        return Self{real: SIMD[type, width].splat(real),
                    imag: SIMD[type, width].splat(imag)}
  
    fn conjugate(self) -> Self:
        """Negates the imaginary component of `self` to produce the complex conjugate.
        
        Returns:
            The complex conjugate of `self` as a `ComplexSIMD`.
        """
        return Self(self.real, -self.imag)
    
    fn squared_abs(self) -> Self._type:
        """Computes the square of the absolute value of `self`, i.e. `self.real**2 + self.imag**2`.
        
        Returns:
            A real `SIMD` representing the square of the absolute value of `self`.
        """
        return self.real*self.real + self.imag*self.imag
    
    fn abs(self) -> Self._type:
        """Computes the absolute value of `self`, i.e. `sqrt(self.real**2 + self.imag**2)`.
        
        Returns:
            A real `SIMD` representing the absolute value of `self`.
        """
        return hypot(self.real, self.imag)
    
    fn __abs__(self) -> Self:
        """Computes the absolute value of `self`, i.e. `sqrt(self.real**2 + self.imag**2)`.
        
        Returns:
            A `ComplexSIMD` representing the absolute value of `self`. The imaginary component is set to zero.
        """
        return Self(self.abs(), 0)
    
    fn arg(self) -> Self._type:
        """Computes the argument of `self`, calculated as `atan2(self.imag, self.real)`. The argument therefore
        always lies in the range (-π, π].
        
        Returns:
            A `SIMD` representing the argument value of `self`.
        """
        return atan2(self.imag, self.real)

    fn cast[target: DType](self) -> ComplexSIMD[target, width]:
        """Casts `self` to the target datatype."""
        return ComplexSIMD[target, width](self.real.cast[target](), self.imag.cast[target]())
    
    fn to_polar(self) -> Tuple[Self._type, Self._type]:
        """Converts the complex values in `self` to polar coordinates using `self.abs()` and `self.arg()`.
        
        Returns:
            A `Tuple` of `SIMDs` containg the values of `self` in polar form.
        """
        return Tuple(self.abs(), self.arg())
    
    @staticmethod
    fn from_polar(r: Self._type, theta: Self._type) -> Self:
        """Converts a pair of `SIMDs` representing the polar form coordinates of a set of complex values
        to a `ComplexSIMD` in standard cartesian form.

        Args:
            r: The distance from the origin, interpreted as the absolute value of the cartesian form.
            theta: The angle that `r` makes from the abscissa, interpreted as the argument of the cartesian form.
        
        Returns:
            A `ComplexSIMD` containing cartesian complex values converted from the polar form values of `r` and `theta`.
        
        """
        return Self(r*cos(theta), r*sin(theta))
    
    @staticmethod
    fn i() -> Self:
        """A quick constructor for the imaginary number i. i.e. A `ComplexSIMD` where all real values are zero, and all
        imaginary values are one."""
        return Self(0, 1)





