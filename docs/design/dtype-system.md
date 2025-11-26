# Dtype system

`DType` is an enum of the 15 element types NumJVM knows. A dtype describes storage: size, alignment, category and memory
layout. Elements are always stored in the platform's native byte order; byte order belongs to external formats such as
`.npy`, not to dtypes.

| DType                     | NumPy                     |  Size | Alignment | Kind | Element access                  | `.npy`                     |
|---------------------------|---------------------------|------:|----------:|------|---------------------------------|----------------------------|
| `BOOL`                    | `bool` (`b1`)             |     1 |         1 | `b`  | yes                             | yes                        |
| `INT8` … `INT64`          | `int8` … `int64`          |   1–8 |    = size | `i`  | yes                             | yes                        |
| `UINT8` … `UINT64`        | `uint8` … `uint64`        |   1–8 |    = size | `u`  | yes (uint64: see below)         | yes                        |
| `FLOAT16`                 | `float16` (`f2`)          |     2 |         2 | `f`  | yes, via `Float.float16ToFloat` | yes                        |
| `BFLOAT16`                | none                      |     2 |         2 | `f`  | yes                             | **no** (`NUMJVM_SPECIFIC`) |
| `FLOAT32`, `FLOAT64`      | `float32`, `float64`      |  4, 8 |    = size | `f`  | yes                             | yes                        |
| `COMPLEX64`, `COMPLEX128` | `complex64`, `complex128` | 8, 16 |      4, 8 | `c`  | **no** (storage only)           | yes                        |

Complex arrays can be created (`zeros`, `astype` from real), copied, viewed, broadcast, printed and saved; element
accessors and the validation kernels reject them with
`DTypeException` until M1.

Client code should use the query methods (`size()`, `alignment()`, `kind()`,
`isInteger()`, `isFloatingPoint()`, ...) rather than switching on constants, so that dtypes added later (strings,
datetimes) do not break it.

## Conversions

Element accessors convert between dtypes and Java primitives:

|                   | rule                                                                                      |
|-------------------|-------------------------------------------------------------------------------------------|
| read as `double`  | any real dtype; bool → 0/1; uint64 as its unsigned value                                  |
| read as `long`    | bool and integers; uint64 returns the raw 64 bits                                         |
| read as `boolean` | any real dtype; non-zero (including NaN) → `true`                                         |
| write `double`    | integers: truncate toward zero, wrap to the width; bool: `!= 0`; floats: round to nearest |
| write `long`      | integers: keep the low bits; floats: round to nearest                                     |

`astype(target)` uses the same rules element by element, through `long` for integer/bool pairs (so 64-bit values
survive) and through `double` otherwise. Real becomes complex with a zero imaginary part; complex to real is refused
rather than silently dropping the imaginary part (NumPy warns and drops it).

Float16 is converted through `float`; bfloat16 rounds to nearest even and keeps NaN quiet.

## uint64

Java has no unsigned 64-bit primitive. `getLong` returns the bits, so values from 2⁶³ upward appear negative;
`getDouble` returns the unsigned value (rounded to double), and
`org.numjvm.dtype.Unsigned` formats or converts bits:

```java
NDArray u = NJ.array(new long[]{-1}).astype(DType.UINT64);
u.

getLong(0);                      // -1
Unsigned.

toString(u.getLong(0));   // "18446744073709551615"
        u.

getDouble(0);                    // 1.8446744073709552E19
```

## NumPy descriptors

`numpyDescriptor()` returns the array-protocol string in native order (`<f8`, `|b1`);
`fromNumpyDescriptor` accepts `<`, `>`, `=` and `|` prefixes and rejects everything NumJVM cannot represent (strings,
objects, datetimes, structured types, bfloat16).

## Type promotion

Not part of M0. M1 introduces promotion rules following a pinned revision of the Python Array API Standard; NumPy's
value-based casting history is not copied.
