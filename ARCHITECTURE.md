# NumJVM architecture

NumJVM is a Java-first numerical library: Java owns the array model, memory management and the numerical loops; native
code is used only where existing implementations give a clear advantage (BLAS from M2; LAPACK remains a follow-up). This document
describes the CPU providers, expression execution and future extension points.

```text
                 Java / Kotlin / Scala callers
                              │
                          NDArray API           org.numjvm, org.numjvm.ndarray
                              │
              ┌───────────────┴──────────────┐
        Ufuncs and reductions          JIT engine (M3)
        org.numjvm.ufunc                 │
              └───────────────┬──────────────┘
                  Type promotion               org.numjvm.dtype (TypePromotion, Casting)
                              │
                  Iteration plans              org.numjvm.iteration (IterationPlan, StridedIterator)
                              │
                  Kernel registry              internal.loops (generated + hand-written loops)
              ┌───────────────┼──────────────┐
         Scalar Java     Optional Vector API (M2)   Optional OpenBLAS GEMM (M2)
              └───────────────┬──────────────┘
                        Memory layer           org.numjvm.memory, internal.memory
                              │
                        MemorySegment          java.lang.foreign
```

## Modules

| Module                | JPMS name            | Depends on    | Contents                                                                    |
|-----------------------|----------------------|---------------|-----------------------------------------------------------------------------|
| `numjvm-core`         | `org.numjvm.core`    | JDK only      | dtypes, memory, NDArray, views, broadcasting, iteration, ufuncs, reductions |
| `numjvm-io`           | `org.numjvm.io`      | core          | `.npy` read, write and memory-map                                           |
| `numjvm-vector` | `org.numjvm.vector` | core, incubating Vector API | optional generated SIMD loops |
| `numjvm-linalg` | `org.numjvm.linalg` | core | Java matmul, optional OpenBLAS LP64 GEMM |
| `numjvm-bench`        | none (not published) | core, io, JMH | benchmarks                                                                  |
| `numjvm-tests-oracle` | none (Python)        | NumPy         | differential test fixtures                                                  |

Core, I/O and Java linalg need no Python, NumPy or native library at runtime.
The vector artifact is optional and requires the incubator module when selected.

## Packages of `org.numjvm.core`

| Package                      | Exported                | Contents                                                                    |
|------------------------------|-------------------------|-----------------------------------------------------------------------------|
| `org.numjvm`                 | yes                     | `NJ` (factories, scopes, comparisons, memory stats), `@Experimental`        |
| `org.numjvm.dtype`           | yes                     | `DType`, `DTypeKind`, `Unsigned`, `Complex`, `TypePromotion`, `Casting`    |
| `org.numjvm.ndarray`         | yes                     | `NDArray`, `Shape`, `Strides`, `ByteExtent`, `ArrayFlags`, `Slice`, `Order` |
| `org.numjvm.shape`           | yes                     | `Broadcasting`                                                              |
| `org.numjvm.memory`          | yes                     | `NJScope`, `MemoryOwner`, `MemoryStats`                                     |
| `org.numjvm.iteration`       | yes, experimental       | `IterationPlan`, `StridedIterator`, `IndexIterator`                         |
| `org.numjvm.ufunc`           | yes                     | `Ufuncs`, `BinaryUfunc`, `UnaryUfunc`, `Reductions`, `Rounding`             |
| `org.numjvm.exception`       | yes                     | `NumJVMException` and subclasses                                            |
| `org.numjvm.internal`        | to trusted I/O/linalg modules | `SharedSecrets`, `Factories`, `HalfFloat`                                   |
| `org.numjvm.internal.loops`  | to vector provider                      | `KernelRegistry`, generated and complex loops, `Pairwise`, `ScalarMath`     |
| `org.numjvm.internal.memory` | to trusted I/O/linalg modules | `Buffer`, allocators, tracker                                               |

`Buffer` stays internal: exposing the full writable segment would let callers bypass the read-only flag of views.
Internal callers in other packages reach non-public members through `SharedSecrets`, the pattern the JDK uses for the
same problem.

## Key decisions

| Topic            | Decision                                                                                 | Record                                            |
|------------------|------------------------------------------------------------------------------------------|---------------------------------------------------|
| Memory ownership | One owner per buffer; arrays and views hold the buffer strongly                          | [memory model](docs/design/memory-model.md)       |
| Heap vs native   | `long[]` heap buffers ≤ 1 MiB, 64-byte aligned native above; scoped arrays always native | [ADR-001](docs/adr/ADR-001-allocation-model.md)   |
| `keep()`         | Always a copy                                                                            | ADR-001                                           |
| Strides          | Bytes, `long`, may be negative or zero                                                   | [NDArray model](docs/design/ndarray-model.md)     |
| Validation       | Byte extent checked once per array; accessors check only indices                         | ADR-002                                           |
| Element access   | Typed, rank-specialised accessors with constant unaligned layouts                        | [ADR-002](docs/adr/ADR-002-element-access-api.md) |
| Iteration        | External inner loop over an odometer of outer axes                                       | [ADR-003](docs/adr/ADR-003-iterator-shape.md)     |
| Iteration plans  | Axes reordered by stride and coalesced before every loop, as NumPy's `nditer`             | [ADR-004](docs/adr/ADR-004-iteration-plans.md)    |
| Promotion        | NumPy 2 / Array API v2025.12 lattice; Java scalars are weak; first safe loop             | [ADR-005](docs/adr/ADR-005-type-promotion.md)     |
| Loops            | Generated from templates at build time, keyed by operation and input dtypes             | [ufuncs](docs/design/ufuncs.md)                   |
| Accuracy         | Exact, except transcendental functions within 4 ulp of NumPy (D6)                        | [ufuncs](docs/design/ufuncs.md)                   |
| Broadcasting     | Zero-copy, read-only views                                                               | [broadcasting](docs/design/broadcasting.md)       |
| Dtypes           | Runtime dtype, no `NDArray<T>`; native byte order only                                   | [dtype system](docs/design/dtype-system.md)       |
| Reshape          | `reshape` may copy; `reshapeView` never does                                             | NDArray model                                     |
| I/O              | `.npy` v1–v3 in M0, no pickles; `.npz` later                                             | `org.numjvm.io` Javadoc                           |
| Compatibility    | Every behaviour tagged `ARRAY_API`, `NUMPY_COMPATIBILITY` or `NUMJVM_SPECIFIC`           | [oracle](numjvm-tests-oracle/README.md)           |

## Testing

- **Unit tests** (JUnit 5) for every public operation and failure mode.
- **Property tests** (jqwik) over generated shapes, strides and arrays: extents equal enumerated offsets, transposing
  twice is the identity, reshape preserves values, broadcasting preserves source values, iteration visits the enumerated
  offsets.
- **NumPy oracle**: 4,934 cases replayed against NumPy's results, including every ufunc on every dtype in five layouts
  and Hypothesis-drawn cases; NumPy then reads every file NumJVM wrote. NumPy-generated tables additionally check 864
  slice and 610 reshape cases.
- **Benchmarks** (JMH) check that metadata operations are size-independent and that element access and iteration do not
  allocate.

## Where later milestones attach

- **CPU fast paths**: Vector API kernels on supported buffers; BLAS through FFM downcalls on
  the same segments.
- **M3 JIT**: traces `NDArray` expressions and generates fused kernels with the ClassFile API; the array model does not
  change.
- **M4 ecosystem**: Arrow and DLPack exchange buffers; this is where an advanced, explicitly unsafe segment API would be
  introduced.


M2's implemented provider boundaries, conversion leases, numerical contracts and
native packing are described in [CPU execution](docs/design/cpu-backends.md) and
[ADR-006](docs/adr/ADR-006-cpu-backends.md). Thread pools and LAPACK remain follow-ups.
