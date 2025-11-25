# Changelog

All notable changes to NumJVM are listed here. The project follows
[Semantic Versioning](https://semver.org/); until 1.0 every minor version may change the API.

## [0.1.0-alpha] - 2025-11-25

Milestone M0: the core array and memory foundation.

### Added

- **Dtypes**: bool, int8–64, uint8–64, float16, bfloat16, float32, float64,
  complex64, complex128 with size, alignment, kind, memory layout and NumPy
  descriptors. Complex arrays are storage-only.
- **Memory**: `long[]`-backed heap buffers and 64-byte aligned native buffers chosen by a
  1 MiB threshold (`-Dnumjvm.memory.heapThreshold`); `NJScope` with confined and shared
  modes; `keep()`; native memory accounting in `NJ.memoryStats()` with an opt-in soft
  limit (`-Dnumjvm.memory.nativeSoftLimit`).
- **NDArray**: byte strides (negative and zero), 0-D and empty arrays, invariants
  validated once at construction, `ArrayFlags`, NumPy-style `toString()`.
- **Element access**: `getDouble/getLong/getBoolean` and setters for rank 0–3 and any
  rank via `long[]`, allocation-free, with documented conversions; `Unsigned` helpers
  for uint64.
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
- **I/O** (`numjvm-io`): `.npy` read, write and memory-map; formats 1.0–3.0, both byte
  orders, C and Fortran order, input limits, no pickles. Output is byte-identical to
  `np.save`.
- **Testing**: JUnit 5 and jqwik property tests, a 103-case NumPy oracle replayed by the
  build, and NumPy reading every file NumJVM writes in CI.
- **Benchmarks**: JMH suite for the 12 M0 benchmarks and a NumPy comparison script.
- **Documentation**: architecture, design documents, ADR-001 to ADR-003, benchmark report.

### Known limitations

- No elementwise engine, dtype promotion or reductions beyond the two validation
  kernels (M1).
- Kernels are scalar Java loops: 2–4× slower than NumPy on contiguous data and up to
  13× slower on transposed data, which needs loop reordering (M1) and SIMD (M2).
- `@Experimental` APIs (`iteration`, `kernels`) will change in M1.

[0.1.0-alpha]: https://github.com/stamich/NumJVM/releases/tag/v0.1.0-alpha
