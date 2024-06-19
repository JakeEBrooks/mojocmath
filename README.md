## A Mojo Math Library for Complex Numbers

- Expands the `ComplexSIMD` type to include complex operations with other `ComplexSIMDs` and real operations with other `SIMDs`.
- Also adds fundamental functions for calculating absolute values, arguments, and converting to/from polar form
- Adds a library of math functions that operate on `ComplexSIMDs`, which includes:
    - Trigonometric Functions: `csin`, `ccos`, `casin`, `cacosh`, etc
    - Exponential Functions: `cexp`, `csquare`, `cpow`, etc
    - Logarithmic Functions: `clog`, `clog10`, `clogb`, etc
    - General Functions, such as `csqrt`

## Example

```mojo
from cmath import ComplexSIMD, ComplexScalar, csqrt

fn main():
    var c = ComplexSIMD[DType.float64, 2].i()

    print("i: ", c)

    c[0] = ComplexScalar(-4, -0.)
    c[1] = ComplexScalar(-4, 0.)

    print("Branch Cuts are handled: ", csqrt(c))
```

```
i:  [0.0, 0.0] [1.0, 1.0]i
Branch Cuts are handled:  [0.0, 0.0] [-2.0, 2.0]i
```

## Acknowledgements
Created using a variety of sources including the [Rust standard library](https://docs.rs/num-complex/0.4.6/src/num_complex/lib.rs.html) implementation of complex mathematics. Written in [Mojo](https://github.com/modularml/mojo/tree/main), which is currently being developed by [Modular](https://www.modular.com/).