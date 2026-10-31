# Changelog

All notable changes to NumJVM are listed here. The project follows
[Semantic Versioning](https://semver.org/); until 1.0 every minor version may change the API.

## [0.4.0-alpha] - Unreleased

Milestone M3 draft candidate: explicit expression tracing and experimental block
and generated CPU execution. Reference performance and release checks remain open.

### Added

- Optional `numjvm-jit` artifact with float32/64 expression graphs, bounded block
  scratch, generated kernels, reusable contexts and cache diagnostics.
- Expression examples, focused correctness tests, staged consumer checks and
  exploratory benchmark data. See [M3 status](docs/milestones/M3-status.md).

## [0.3.0-alpha] - Unreleased

Milestone M2 implementation candidate. Release performance and remote quality gates
remain open; see [implementation evidence](docs/milestones/M2-status.md).

### Added

- Optional `numjvm-vector` artifact with generated float32/64 and int32/64 kernels,
  selected reductions, immutable backend configuration and execution diagnostics.
  Scalar fallback and the existing pairwise numerical tree remain available.
- `numjvm-linalg`: real float32/64 `Linalg.matmul`, vector/matrix promotions,
  broadcast batches, Java fallback, scoped results, exact output dtype/shape,
  alias-safe copyback and arbitrary valid input strides.
- Optional Linux x86-64 OpenBLAS LP64 CBLAS float32/64 GEMM through checked FFM
  bindings, explicit native access, packing and provider/dispatch diagnostics.
- Backend CI definitions, direct parity/dispatch/lifetime tests, NumPy and
  high-precision matmul checks, independent classpath/JPMS consumers and examples.
- Separate allocation/output/packing benchmarks, native counter probe and retained
  three-fork matrix measurements with uncertainty.

### Changed

- Mixed binary conversions use bounded per-call scratch with a global active budget;
  hazardous aliases retain snapshot fallback.
- Narrow integer sum/prod read directly into widened accumulators. Integer mean
  avoids full-input casts, retaining cast-layout arithmetic order for strided inputs.
- Compatible real reduction outputs execute directly after lifecycle validation.
- M2 scope excludes worker pools, LAPACK, dot/vecdot, complex matmul and additional
  native ABIs. Those require separately planned follow-ups.

## [0.2.0-alpha] - 2026-01-06

Milestone M1: the elementwise engine, and the quality baseline it was built under.

### Added

- **Ufuncs** (`org.numjvm.ufunc`): 41 universal functions as constants of `Ufuncs`, with shortcuts on `NJ` and
  `NDArray`: arithmetic (`add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `power`, `negative`,
  `positive`, `absolute`, `conjugate`, `sign`, `square`), extrema (`maximum`, `minimum`, `fmax`, `fmin`), comparisons,
  logical and bitwise operations, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `floor`, `ceil`, `trunc`, `rint`, and
  `round(x, decimals)`. Every ufunc broadcasts, accepts weak Java `long`, `double` and `Complex` scalars, and takes an
  `out=` array; `addInPlace` and friends operate in place.
- **Type promotion** (ADR-005): NumPy 2 / Array API v2025.12 lattice, `NJ.resultType`, `NJ.canCast` with the five
  casting kinds, weak scalars (NEP 50), exact comparisons of signed integers with uint64.
- **Reductions**: `sum` (pairwise, bit-identical to NumPy), `prod`, `max`, `min`, `mean`, `any`, `all` over axis lists
  with `keepdims`, and `BinaryUfunc.reduce` for any reorderable ufunc.
- **Iteration plans** (ADR-004): axes reordered by stride and coalesced, as NumPy's `nditer`; inner loops have
  contiguous, broadcast, reduction and strided paths. Transposed and strided operands are read sequentially.
- **Kernel registry**: about 650 inner loops, most generated at build time from templates (`numjvm-core/src/codegen`),
  registered in NumPy's loop order; `astype` runs on the generated cast loops.
- **Complex arithmetic**: `Complex` value type, `getComplex`/`setComplex`, `real()`/`imag()` views, NumPy-style
  printing; add, subtract, multiply, divide (Smith), absolute, conjugate, square, sign, exp, log, sqrt, rint, equality,
  sums and means.
- **Results** are laid out in the operands' memory order (NumPy's `order='K'`) and live in the scope of a scoped operand.
- **Accuracy contract** (decision D6): exact results except `power`, `exp`, `log`, `sin`, `cos`, `tanh` (≤ 4 ulp) and
  complex `divide`, `absolute`, `sign` (≤ 2 ulp).
- **Oracle**: 4,934 cases (from 103), including every ufunc × dtype × five layouts and 280 Hypothesis-drawn cases; a
  4 MB fixture budget; ULP-tolerant comparison in Java and Python.
- **Quality**: SonarCloud CI analysis with a blocking quality gate, an 80 % line-coverage floor (98.7 % reached), CI
  actions pinned to commit SHAs with least-privilege permissions.
- **Benchmarks**: JMH suites for ufuncs and reductions with NumPy counterparts and a comparison script; results in
  `docs/benchmarks/M1.md`.
- **Documentation**: ADR-004, ADR-005, design documents for ufuncs and type promotion, M1 retrospective.

### Changed

- `astype` and copies convert through the registry's cast loops; float64 → float16/bfloat16 rounds once, directly from
  the double.
- Complex arrays are no longer storage-only: `setDouble`/`setLong` store the real part, `getBoolean` tests both parts.
- `SharedSecrets` accessors throw `IllegalStateException` instead of returning `null` before initialisation.

### Removed

- `org.numjvm.kernels.ValidationKernels` and the `org.numjvm.kernels` package: use `NJ.add`, `NJ.sum` and the other
  ufuncs and reductions.

### Known limitations

- Inner loops are scalar Java; NumPy's SIMD loops are faster on contiguous data (see `docs/benchmarks/M1.md`). Vector
  API kernels are M2 work.
- Complex ordering (`less`, `maximum`, ...), complex `power` and complex trigonometric functions are not implemented.
- No `where=` masks, `accumulate`, `outer` or generalized ufuncs.

## [0.1.0-alpha] - 2025-11-25

Milestone M0: the core array and memory foundation.

### Added

- **Dtypes**: bool, int8–64, uint8–64, float16, bfloat16, float32, float64, complex64, complex128 with size, alignment,
  kind, memory layout and NumPy descriptors. Complex arrays are storage-only.
- **Memory**: `long[]`-backed heap buffers and 64-byte aligned native buffers chosen by a 1 MiB threshold
  (`-Dnumjvm.memory.heapThreshold`); `NJScope` with confined and shared modes; `keep()`; native memory accounting in
  `NJ.memoryStats()` with an opt-in soft limit (`-Dnumjvm.memory.nativeSoftLimit`).
- **NDArray**: byte strides (negative and zero), 0-D and empty arrays, invariants validated once at construction,
  `ArrayFlags`, NumPy-style `toString()`.
- **Element access**: `getDouble/getLong/getBoolean` and setters for rank 0–3 and any rank via `long[]`,
  allocation-free, with documented conversions; `Unsigned` helpers for uint64.
- **Factories**: `zeros`, `ones`, `empty`, `full`, `array` (from Java arrays), `scalar`,
  `arange`, on `NJ` and on `NJScope`.
- **Views**: `transpose`, `permuteDims`, `swapAxes`, `slice` (NumPy basic indexing),
  `reshape`/`reshapeView`/`ravel` (NumPy's no-copy rule), `broadcastTo` (read-only),
  `sharesMemoryWith`.
- **Copies**: `copy(Order)`, `astype(DType)` with NumPy's unsafe casting rules.
- **Broadcasting**: `Broadcasting.broadcastShapes` and `broadcastStrides`.
- **Iteration**: `StridedIterator` (multi-operand, external inner loop) and
  `IndexIterator`.
- **Kernels**: experimental `ValidationKernels.add` and `sumFloat64` (pairwise).
- **I/O** (`numjvm-io`): `.npy` read, write and memory-map; formats 1.0–3.0, both byte orders, C and Fortran order,
  input limits, no pickles. Output is byte-identical to
  `np.save`.
- **Testing**: JUnit 5 and jqwik property tests, a 103-case NumPy oracle replayed by the build, and NumPy reading every
  file NumJVM writes in CI.
- **Benchmarks**: JMH suite for the 12 M0 benchmarks and a NumPy comparison script.
- **Documentation**: architecture, design documents, ADR-001 to ADR-003, benchmark report.

### Known limitations

- No elementwise engine, dtype promotion or reductions beyond the two validation kernels (M1).
- Kernels are scalar Java loops: 2–4× slower than NumPy on contiguous data and up to 13× slower on transposed data,
  which needs loop reordering (M1) and SIMD (M2).
- `@Experimental` APIs (`iteration`, `kernels`) will change in M1.

[0.2.0-alpha]: https://github.com/stamich/NumJVM/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/stamich/NumJVM/releases/tag/v0.1.0-alpha
