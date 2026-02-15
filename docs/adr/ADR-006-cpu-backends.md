# ADR-006: optional CPU backends and M2 arithmetic contracts

Status: accepted for experimental 0.3.0-alpha implementation.

## Boundaries and configuration

Core resolves dtype, shape, ownership and aliases before selecting a backend.
An internal service interface supplies loops; only trusted optional modules receive
qualified JPMS access. No incubator type or raw segment enters the public API.
Core-only consumers must run without resolving jdk.incubator.vector. Both classpath
service files and JPMS `provides` declarations are required.

`numjvm.simd=auto|off|required` is read once at backend initialization. Required
means a provider must be usable, not that every operation is vectorized. Scalar
fallback remains valid for unsupported dtypes/layouts/runs and reductions whose
arithmetic tree cannot be preserved. Invalid configuration is an error. Diagnostics
separate provider availability from actual accelerated execution. Discovery errors
are reported, never swallowed as successful acceleration.

`numjvm.linalg=auto|java|openblas` follows the same availability rule. A forced
OpenBLAS setting rejects missing/incompatible providers. Empty operations require
no downcall; dimensions outside LP64 limits use Java before any native execution.
Auto mode may choose Java for small matrices. Diagnostics name the actual path.
Native execution failures are propagated; never retry a possibly partial write.

## Existing arithmetic

Preserve promotion, output casting, overflow, exceptions, layout and scope rules.
SIMD must preserve finite bits and signed zeros against scalar execution; NaN
classification is required, payload identity is not. Do not introduce FMA into
existing elementwise operations. Floating sum retains eight logical accumulators,
128-element blocks and recursive combination independent of physical species.
Integer mean converts each element to float64 before summation, preserving the
cast-then-reduce traversal/grouping; summing integers before conversion is invalid.

## Matrix arithmetic

Float32 inputs accumulate in float32; mixed float32/64 promotes to float64 before
multiplication. The simple Java reference uses sequential multiply then add.
Optimized matrix kernels and BLAS may use FMA and different summation orders.
For finite values without intermediate overflow, test componentwise absolute error
against a high precision reference with a bound based on gamma_k = k*u/(1-k*u),
where u is half dtype epsilon, scaled by sum(abs(a_i*b_i)); use an explicitly
reported small absolute floor for subnormals. Compute reference and bound with
higher precision when machine arithmetic could overflow. Test exceptional inputs
separately: NaN classification, infinity and cancellation; cross-backend signed
zero identity is not promised for matmul. Do not silently enable flush-to-zero.
Rank zero is invalid; 1-D promotions, batch broadcasting and zero contraction
follow matmul semantics. Unsupported dtypes fail rather than silently cast.

## Memory and errors

Results use the first live operand scope, otherwise automatic storage. User out
must be exact shape/dtype and writable. Alias hazards compute into independent
storage before copyback. Scope/thread, shape, dtype, size and native argument
validation precede output modification. Once arithmetic/copyback starts, unexpected
execution failures may leave partial out; output updates are not transactional.

Scratch leases cannot escape, overlap a live lease, or own returned arrays.
Bound total active scratch process-wide, including nested/concurrent callers,
and retain no thread-local pool initially. Release even on exceptions. Pooling is
an evidence-driven follow-up. Alias snapshots and matrix copyback allocations are
explicit exceptions to bounded conversion scratch.

## Delivery

Validate scalar improvements, optional SIMD and multiplication independently.
Keep tests and consumer smoke checks near each feature, followed by release-wide
verification. Thread pools, LAPACK and complex matmul are deferred and must be
replanned before M3; they are not implied by completion of this milestone.

The implementation details and supported acceleration matrix are recorded in
[CPU execution](../design/cpu-backends.md). Automatic GEMM dispatch includes packing
cost: current provisional minimum dimensions are 32 for native-compatible operands
and output, 128 otherwise; see the retained three-fork experiment before tuning.
