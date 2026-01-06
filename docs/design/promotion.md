# Type promotion

Every operation with more than one input needs a result dtype. NumJVM's rules are those of NumPy 2 and the Python Array
API Standard (revision v2025.12), recorded in [ADR-005](../adr/ADR-005-type-promotion.md). This page is the practical
summary; the ADR has the reasoning and the profile tags.

## Arrays with arrays

`NJ.resultType(a, b)` (and `TypePromotion.resultType`) gives the common dtype:

| Operands | Result | Example |
|----------|--------|---------|
| same dtype | that dtype | `int8 + int8 → int8` |
| same kind | the wider | `float32 + float64 → float64` |
| signed + unsigned, up to 32 bits | the smallest signed type holding both | `int8 + uint8 → int16`, `int32 + uint32 → int64` |
| `int64 + uint64` | `float64` | |
| bool + anything | the other dtype | `bool + int16 → int16` |
| integer + float | a float wide enough for both | `uint8 + float16 → float16`, `int16 + float16 → float32`, `int32 + float32 → float64` |
| real + complex | complex of the promoted real part | `float64 + complex64 → complex128`, `int16 + complex64 → complex64` |
| bfloat16 + float16 | `float32` | (`ml_dtypes`) |

With three or more dtypes, NumPy's order applies, not a left-to-right fold: `resultType(int16, uint16, float32)` is
`float32`.

The ufunc then picks a loop for the common dtype; for most ufuncs the result has that dtype. Exceptions follow
NumPy: comparisons give bool, `divide` of integers gives float64, `sqrt` of int8 gives float16 (the first float loop
int8 converts to safely), `absolute` of complex gives the real dtype.

## Arrays with Java scalars

Java `long`, `double` and `Complex` arguments are *weak*, like Python scalars in NumPy 2 (NEP 50): their kind counts,
their width does not.

| Array dtype | `long` | `double` | `Complex` |
|-------------|--------|----------|-----------|
| bool | int64 | float64 | complex128 |
| integer | the array's dtype (out of range: `ValueException`) | float64 | complex128 |
| float16, bfloat16, float32 | the array's dtype | the array's dtype | complex64 |
| float64 | float64 | float64 | complex128 |
| complex | the array's dtype | the array's dtype | the array's dtype |

So `NJ.multiply(float32Array, 2.5)` stays float32 and `NJ.add(uint8Array, 1L)` stays uint8. A 0-D array is not weak:
`NJ.add(uint8Array, NJ.scalar(300L))` is int64. Comparisons with an out-of-range `long` (`uint8Array < 300`) are
computed exactly instead of failing.

## Casting

`NJ.canCast(from, to, casting)` answers NumPy's `np.can_cast` for the five casting kinds:

| Casting | Allowed |
|---------|---------|
| `NO`, `EQUIV` | the same dtype |
| `SAFE` | when no value can change: `resultType(from, to) == to` |
| `SAME_KIND` | `SAFE`, or within the order bool < unsigned < signed < float < complex (float64 → float32 is allowed) |
| `UNSAFE` | always |

Ufunc outputs (`out=`, `addInPlace`) require `SAME_KIND`: an int8 array can receive an int8 or int16 sum, but adding a
float64 array into it is a `DTypeException`. `astype` is always `UNSAFE`, as in NumPy, except that complex to real
(other than bool) is refused rather than dropping the imaginary part silently.
