# NumJVM architecture

NumJVM is a Java-first numerical library: Java owns the array model, memory management and the numerical loops; native
code is used only where existing implementations give a clear advantage (BLAS and LAPACK from M2). This document
describes the state after milestone M0 (0.1.0-alpha) and where later milestones attach.

```text
                 Java / Kotlin / Scala callers
                              │
                          NDArray API           org.numjvm, org.numjvm.ndarray
                              │
              ┌───────────────┴──────────────┐
         Eager engine (M1)              JIT engine (M3)
              └───────────────┬──────────────┘
                     Iteration engine          org.numjvm.iteration      ◄─ M0: StridedIterator
                              │
                       Kernel registry (M1)    org.numjvm.kernels        ◄─ M0: validation kernels
              ┌───────────────┼──────────────┐
         Scalar Java     Vector API (M2)   Native BLAS/LAPACK (M2)
              └───────────────┬──────────────┘
                        Memory layer           org.numjvm.memory, internal.memory
                              │
                        MemorySegment          java.lang.foreign
```

## Modules

| Module                | JPMS name            | Depends on    | Contents                                                                    |
|-----------------------|----------------------|---------------|-----------------------------------------------------------------------------|
| `numjvm-core`         | `org.numjvm.core`    | JDK only      | dtypes, memory, NDArray, views, broadcasting, iteration, validation kernels |
| `numjvm-io`           | `org.numjvm.io`      | core          | `.npy` read, write and memory-map                                           |
| `numjvm-bench`        | none (not published) | core, io, JMH | benchmarks                                                                  |
| `numjvm-tests-oracle` | none (Python)        | NumPy         | differential test fixtures                                                  |

Neither library module needs Python, NumPy or native libraries at run time.

## Packages of `org.numjvm.core`

| Package                      | Exported                | Contents                                                                    |
|------------------------------|-------------------------|-----------------------------------------------------------------------------|
| `org.numjvm`                 | yes                     | `NJ` (factories, scopes, comparisons, memory stats), `@Experimental`        |
| `org.numjvm.dtype`           | yes                     | `DType`, `DTypeKind`, `Unsigned`                                            |
| `org.numjvm.ndarray`         | yes                     | `NDArray`, `Shape`, `Strides`, `ByteExtent`, `ArrayFlags`, `Slice`, `Order` |
| `org.numjvm.shape`           | yes                     | `Broadcasting`                                                              |
| `org.numjvm.memory`          | yes                     | `NJScope`, `MemoryOwner`, `MemoryStats`                                     |
| `org.numjvm.iteration`       | yes, experimental       | `StridedIterator`, `IndexIterator`                                          |
| `org.numjvm.kernels`         | yes, experimental       | `ValidationKernels` (`add`, `sumFloat64`)                                   |
| `org.numjvm.exception`       | yes                     | `NumJVMException` and subclasses                                            |
| `org.numjvm.internal`        | to `org.numjvm.io` only | `SharedSecrets`, `Factories`, `HalfFloat`                                   |
| `org.numjvm.internal.memory` | to `org.numjvm.io` only | `Buffer`, allocators, tracker                                               |

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
- **NumPy oracle**: 103 cases replayed against NumPy's results; NumPy then reads every file NumJVM wrote.
  NumPy-generated tables additionally check 864 slice and 610 reshape cases.
- **Benchmarks** (JMH) check that metadata operations are size-independent and that element access and iteration do not
  allocate.

## Where later milestones attach

- **M1 elementwise engine**: replaces `ValidationKernels` with a kernel registry, dtype promotion, ufuncs and
  reductions. `StridedIterator` gains dimension coalescing and loop reordering behind the same interface.
- **M2 fast paths**: Vector API kernels on the 64-byte aligned native buffers; BLAS and LAPACK through FFM downcalls on
  the same segments.
- **M3 JIT**: traces `NDArray` expressions and generates fused kernels with the ClassFile API; the array model does not
  change.
- **M4 ecosystem**: Arrow and DLPack exchange buffers; this is where an advanced, explicitly unsafe segment API would be
  introduced.
