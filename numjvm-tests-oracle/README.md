# NumPy oracle

NumPy is NumJVM's reference implementation. This directory holds the generator, the committed fixtures, and the script
that checks NumJVM's output with NumPy.

| File                                           | Purpose                                                                                   |
|------------------------------------------------|-------------------------------------------------------------------------------------------|
| `generate_cases.py`                            | Builds input arrays, runs every case through NumPy, writes `fixtures/`                    |
| `fixtures/manifest.json`                       | One entry per case: id, compatibility tag, operation, inputs, parameters, expected result |
| `fixtures/in_*.npy`, `fixtures/expected_*.npy` | Inputs and expected array results                                                         |
| `verify_results.py`                            | Loads the arrays NumJVM wrote and compares them with NumPy's results                      |

Every case carries one tag:

- `ARRAY_API`: behaviour defined by the Python Array API Standard.
- `NUMPY_COMPATIBILITY`: NumPy behaviour beyond the standard that NumJVM follows (printing, contiguity flags, unsafe
  casts).
- `NUMJVM_SPECIFIC`: a deliberate difference, checked against NumJVM's own rule.

## Expectations

A case's `expect` holds one of:

| Key | Meaning |
|-----|---------|
| `file`, `shape`, `dtype`, `fortran` | an array result, stored as `expected_<id>.npy` (or shared with an identical result) and compared byte for byte |
| `value` | a scalar result (sums), compared with a relative tolerance of 10⁻¹² |
| `dtype_name` | a dtype (`result_type`) |
| `can_cast` | `np.can_cast(from, to, casting)` for every target dtype |
| `repr`, `flags`, `shape` | printing, contiguity flags, a broadcast shape |
| `error` | NumJVM must throw a `NumJVMException` |

## The ufunc matrix

Besides hand-picked cases, `generate_cases.py` runs every ufunc on every NumPy dtype (bool, the eight integer types,
float16/32/64, complex64/128) in five layouts of the same logical 3×4 operands (case ids `matrix_<ufunc>_<dtype>_<layout>`):

| Layout | Operands as NumJVM sees them                                           |
|--------|------------------------------------------------------------------------|
| `C`    | C-contiguous arrays                                                    |
| `F`    | Fortran-ordered arrays (read from `.npy` files with `fortran_order`)   |
| `T`    | transposed views of C-contiguous 4×3 arrays                            |
| `S`    | strided views: every other column of 3×8 arrays                        |
| `B`    | broadcasting: a row against the 3×4 array (unary: a row broadcast to 3×4) |

The operands hold each dtype's edge values (extremes, signed zeros, infinities, NaN) plus a few small values. Results
must match in value, dtype and memory order: a Fortran-ordered or transposed input gives a Fortran-ordered result, as
NumPy's `order='K'`. Combinations NumPy supports and NumJVM 0.2 does not (complex ordering comparisons,
maximum/minimum, power and trigonometric functions) are listed in `NOT_IN_NUMJVM` and expect an error, tagged
`NUMJVM_SPECIFIC`.

## Property cases

`property_examples()` uses [Hypothesis](https://hypothesis.readthedocs.io) to draw a further set of cases (ids
`prop_*`): binary ufuncs on random dtype pairs and mutually broadcastable shapes of up to three axes, half of them
through strided views, and reductions (sum, prod, max, min, mean, any, all) over random shapes, axis subsets,
`keepdims` and transposed views. Values are arbitrary, so NaN, infinities, subnormals, signed zeros and the extremes of
each dtype appear wherever Hypothesis puts them.

The draws are fixtures like any other: Hypothesis runs derandomized and only in its generation phase, so the examples
depend on the strategies, NumPy and the Hypothesis version alone. `requirements.txt` pins that version; a different
one draws different examples and `--check` reports the fixtures as out of date. To widen the search, raise
`PROPERTY_SEED_EXAMPLES` and regenerate.

## Units in the last place

Most expected results are compared exactly. Where NumPy and NumJVM legitimately round differently (decision D6),
a case carries `"ulp": n` in its parameters and every element may differ from NumPy's by at most `n` units in the last
place of its dtype:

| Cases                                                       | Tolerance | Why                                                     |
|-------------------------------------------------------------|----------:|---------------------------------------------------------|
| `power`, `exp`, `log`, `sin`, `cos`, `tanh` (real and complex), complex `sqrt` | 4 | `java.lang.Math` against NumPy's SIMD code or glibc |
| complex `divide`, `absolute`, `sign`                        |         2 | libm `hypot`; Smith's formula contracted to FMAs by some compilers |

The distance is measured on the dtype's own grid: values are mapped to a monotonic integer line (sign and magnitude
of the bit pattern, so `-0.0` and `0.0` coincide), and two NaNs are at distance 0. Complex values are compared part by
part. `OracleTest.assertWithinUlps` (Java), `ulp_distance` and `--check` (Python) and `verify_results.py` all use the
same rule, so a fixture regenerated on another CPU passes `--check` as long as it stays within the tolerance.

## Fixture size budget

Array-valued ufunc results with identical contents share one `expected_*.npy` file, so the four layouts of a matrix
row cost one file. The generator fails if inputs, expected files and the manifest together exceed `FIXTURE_BUDGET`
(4 MB); new cases should use small arrays.

## Workflow

`./gradlew test` never needs Python: `OracleTest` in `numjvm-io` replays the committed manifest with the JDK alone. The
`oracle-tests` CI workflow additionally:

1. installs NumPy and runs `generate_cases.py --check`, which regenerates everything into a temporary directory and
   fails if a committed file differs;
2. runs the Java tests, which write NumJVM's array results to `numjvm-io/build/oracle-out`;
3. runs `verify_results.py numjvm-io/build/oracle-out`, so NumPy itself reads every file NumJVM wrote.

To add a case, edit `cases()` in `generate_cases.py`, run it, and commit the fixtures. If the case uses a new operation,
add it to `OracleTest.apply`.
