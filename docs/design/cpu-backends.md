# CPU execution in 0.3.0-alpha

The optional `numjvm-vector` module implements the internal loop-provider service.
Core resolves shape, dtype, scopes and aliases before asking for acceleration.
Core has no dependency on an incubator API. A classpath service registration and
JPMS provider declaration support both consumer styles. Provider construction
forces Vector API linkage; automatic discovery reports an installed but unresolved
provider and retains scalar execution.

`numjvm.simd=auto|off|required` is read once at initialization. Required demands
an available provider, not acceleration of every operation. `BackendInfo` reports
availability/reason separately from the count of executed vector runs. Counts are
process-wide; isolate tests when comparing before/after values.

## Accelerated surface

| Dtype | Operations |
| --- | --- |
| float32/64 | add/subtract/multiply/divide; comparisons; maximum/minimum/fmax/fmin; negative/absolute/square/sqrt |
| int32/64 | add/subtract/multiply; bitwise and/or/xor; comparisons |
| Reductions | float32/64 pairwise sum; int32/64 modular sum; bool any/all; float32/64 min/max on eligible runs |

Contiguous and scalar-broadcast arithmetic runs use scalar tails. Negative-stride
runs and unsupported operations retain scalar kernels. Floating extrema reductions
with zero/NaN lanes retain scalar ordering. Integer public sum widens narrow inputs;
a same-width int32 horizontal kernel is an internal capability, not a change to
public dtype promotion. No existing operation acquires FMA semantics.

Vector kernels are generated from `numjvm-vector/src/codegen` into build outputs.
Pairwise sum preserves eight logical accumulators, 128-element blocks and the scalar
combination tree. Wider species never change that tree. Direct parity tests cover
finite bits, signed zeros, NaN classification, offsets, bounds and fallbacks.

## Conversion and reduction memory

Mixed binary ufuncs convert at most one 64 KiB lease per operation, independent
of input length. Both converted operands share that reservation. Leases are
confined, nested leases are distinct, and no storage is retained after a call.
A 16 MiB process-wide active limit includes concurrent callers; exhaustion raises
MemoryException without blocking or allocating an unbounded fallback. Native
counters include these bytes. Returned arrays never reference scratch storage.
Hazardous output aliases retain independent full snapshots.

Widening integer sum/prod read source values directly. Integer mean converts each
value before summation; non-contiguous inputs use virtual cast-layout strides to
preserve the original reduction runs. Direct reductions accept compatible exact-dtype
C-contiguous non-overlapping outputs. Real float32/64 mean divides in place with
the reference intermediate precision. Other output layouts, casts and incompatible
internal accumulator types use temporary fallback. Lifecycle validation precedes
writing identities to caller memory; unexpected execution failures are not transactional.

## Matmul and native ownership

`numjvm-linalg` exposes `Linalg.matmul(a,b[,out])`. float32/64 only; mixed inputs
produce float64. It implements vector promotion, broadcast batches, arbitrary input
strides, zero dimensions and exact-shape/dtype writable output. Results are C ordered
and use the first operand scope, otherwise automatic storage. Out aliases and strided
outputs compute independently before copyback. Batches execute synchronously.

Java uses blocked kernels with monotonic contraction order and float32 accumulation
for float32 results. BLAS may reorder arithmetic or use FMA. Backend equality is
numerical, not bitwise; [ADR-006](../adr/ADR-006-cpu-backends.md) specifies finite error
bounds and separate exceptional-value expectations. No LAPACK APIs are advertised.

Only Linux x86-64 OpenBLAS LP64 CBLAS is supported natively. Explicit configuration
uses `numjvm.openblas.path`; controlled discovery also checks documented system
locations. Loading verifies `openblas_get_config`, rejects USE64BITINT, and resolves
both GEMM symbols. Library lookup and symbols live for the JVM lifetime. Native-access
flags are required. Every dimension and packed byte extent is validated before a call.
Native-compatible inputs/outputs use their live segments directly; heap, promoted or
strided matrices use synchronous full-matrix packing. This is an explicitly documented
exception to bounded conversion scratch; all packed bytes are accounted and released.

`numjvm.linalg=java|auto|openblas` is also initialized once. Java mode never loads
BLAS. Required OpenBLAS fails when unavailable; it can use Java for empty contractions
or dimensions outside LP64. Auto uses the measured/provisional dispatch policy described
in the benchmark report. `Linalg.nativeProvider()`, `lastBackend()` (calling thread)
and `nativeCalls()` (process-wide) distinguish availability from execution. Native
execution failures propagate without a second attempt. Numerical calls never modify
process-wide BLAS thread settings; callers configure those before launch.

## Validation boundaries

`build` validates Java portability. Optional-provider core/oracle runs use
`-PsimdEnabled=true -PsimdMode=required|off`. Linalg's separate `backendCheck` runs
both Java and required-native tests and enforces combined 80% line coverage with
no native-source exclusions. This avoids making OpenBLAS an installation requirement
for a Java-only build. Performance gates require the designated reference hardware;
shared cloud measurements do not complete those gates.
