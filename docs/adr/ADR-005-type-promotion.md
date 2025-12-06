# ADR-005: Type promotion

- Status: Accepted
- Date: 2025-12-05
- Pins: [Python Array API Standard, revision v2025.12](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
  (decision D5 of the M0 plan, D4 of the 0.2.0 plan)

## Context

Every binary operation needs a result dtype for its two input dtypes, and every operation with an output array needs
to know which conversions are allowed. NumPy's rules changed in NumPy 2 (NEP 50: Python scalars are "weak" and no
longer inspected by value); the Array API Standard defines a subset of them. The oracle compares NumJVM with NumPy,
so where the two sources disagree or the standard is silent, the choice must be explicit and tagged.

## Decision

### 1. The promotion lattice

`result_type(a, b)` follows the pinned Array API revision where it defines a result (`ARRAY_API`), and NumPy 2 where
it does not (`NUMPY_COMPATIBILITY`):

| Operands | Result | Profile |
|----------|--------|---------|
| same dtype | that dtype | ARRAY_API |
| two signed integers, two unsigned integers, two floats, two complex | the wider one | ARRAY_API |
| signed × unsigned integers up to 32 bits | the smallest signed integer holding both (`int8 + uint8 → int16`, `int32 + uint32 → int64`) | ARRAY_API |
| `int64 + uint64` | `float64` | NUMPY_COMPATIBILITY |
| bool × any | the other dtype | NUMPY_COMPATIBILITY |
| integer × float | the smallest float of at least the float's width that holds the integer: `int8/uint8 + float16 → float16`, `int16/uint16 + float16 → float32`, any 32- or 64-bit integer `+ float16/float32 → float64`, `+ float64 → float64` | NUMPY_COMPATIBILITY |
| real × complex | complex of the promoted real component: `float64 + complex64 → complex128`, `int32 + complex64 → complex128`, `int16 + complex64 → complex64` | ARRAY_API for float × complex, NUMPY_COMPATIBILITY otherwise |
| bfloat16 × float16 | `float32` | NUMJVM_SPECIFIC |
| bfloat16 × anything else | as float16 would promote | NUMJVM_SPECIFIC |

bfloat16 is not a NumPy dtype; its rules are those of the `ml_dtypes` package, which NumPy users use for it.

With more than two operands the promotion is not a left-to-right fold. As in NumPy, the operand with the highest NumPy
type number is the starting point (float16 has the highest, 23; bfloat16 ranks above it as a user-defined dtype would)
and the others are promoted into it in order: `result_type(int16, uint16, float32)` is float32, while folding left to
right would give float64. This rule was confirmed against NumPy 2.5 for every triple of the 14 NumPy dtypes.

### 2. Casting rules

`can_cast(from, to, casting)` takes NumPy's five casting kinds:

| Casting | Allowed |
|---------|---------|
| `NO`, `EQUIV` | `from == to` (dtypes have no byte order in NumJVM, so the two are the same) |
| `SAFE` | exactly when `result_type(from, to) == to` |
| `SAME_KIND` | `SAFE`, or when the kind does not decrease in the order bool < unsigned < signed < float < complex: any integer to any float, any float to any float, signed to signed, unsigned to signed or unsigned |
| `UNSAFE` | always |

The default for ufunc outputs (`out=`) is `SAME_KIND`, as in NumPy.

### 3. Java scalars are weak

A Java `long` or `double` argument to an operation (`NJ.add(a, 2.0)`) is a Python-scalar equivalent: its kind matters,
its width does not (NEP 50, Array API "mixing arrays with Python scalars"):

| Array dtype | `long` scalar | `double` scalar |
|-------------|---------------|-----------------|
| bool | int64 | float64 |
| integer | the array's dtype; a value outside its range is a `ValueException`, as NumPy's `OverflowError` | float64 |
| float, complex | the array's dtype | the array's dtype |

Comparisons with an out-of-range `long` (`uint8_array < 300`) are computed exactly instead of failing, as in NumPy 2.

### 4. Loop selection

An operation first promotes its inputs, then uses the first loop in the kernel registry's order for that operation
whose input dtype the promoted dtype can be cast to with `SAFE` casting (NumPy's default type resolver). Operations
with special resolution rules say so, for example true division of integers, which gives `float64`.

## Consequences

- Positive: every result dtype is tagged with where its rule comes from; the oracle checks the complete
  14 × 14 NumPy table plus `can_cast` for every pair and kind.
- Positive: loop selection by "first safe loop" reproduces NumPy's less obvious results without special cases:
  `sqrt(int8)` is float16, `floor_divide(bool, bool)` is int8.
- Negative: NumPy-only rules (`int64 + uint64 → float64`, integer × float) are part of NumJVM's contract. A future
  Array API revision that defines them differently will need a decision, not just an upgrade.
- Negative: weak scalars mean `NJ.add(uint8Array, 300L)` fails while `NJ.add(uint8Array, NJ.scalar(300L))` (an int64
  0-D array, which is not weak) gives int64. This matches NumPy 2 and is documented on the operations.
