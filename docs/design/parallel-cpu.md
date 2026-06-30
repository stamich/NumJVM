# Opt-in parallel CPU execution

`NJ.parallel(workers)` creates an experimental, reusable CPU context. Existing
NumJVM calls remain single-worker. Always close the context:

```java
try (var pool = NJ.parallel(4)) {
    var a = NJ.ones(DType.FLOAT32, 65536);
    var b = NJ.ones(DType.FLOAT32, 65536);
    var out = NJ.zeros(DType.FLOAT32, 65536);
    pool.apply(Ufuncs.ADD, a, b, out);
    var total = pool.sum(out);
    System.out.println(pool.lastDecision());
}
```

The worker limit is 1–256. At most that many worker tasks are submitted per
operation. Calls on one context are serialized; separate contexts can run
concurrently. Workers are reused until `close()`, which waits for active work.
Closing from a context's worker is rejected. Nested work on the same context
runs on that worker without submitting further tasks. A worker cannot start an
operation on a different context or close any context; those calls are rejected
to avoid cross-pool deadlocks. Worker failures propagate
after all submitted tasks finish. An interrupted caller waits for workers,
restores its interrupt flag and receives an `IllegalStateException`. Unexpected
failures can leave preallocated output partially written; operations are not
transactional. Do not mutate inputs or outputs concurrently.

## Supported operations and fallbacks

The `apply` overloads accept a unary or binary ufunc and a preallocated output.
Parallel execution is currently limited to float32/64 C-contiguous arrays with
matching shapes and dtypes, at least 32,768 elements and more than one worker.
The supported unary operations are negative, absolute, square and sqrt; binary
operations are add, subtract, multiply and divide. Safe identical-view in-place
outputs can run in parallel. Shifted overlap, broadcasting, mixed ufunc dtypes,
other layouts and other operations use the existing sequential ufunc, including
its validation and temporary-copy behavior. The threshold is provisional.

For expressions, call `expression.evalInto(pool, out, inputs...)`. Parallel
execution requires C-contiguous inputs of the same shape as a C-contiguous
output, float32/64 dtypes, safe output overlap and at least 32,768 elements.
Mixed float32/64 expressions are eligible. Otherwise the existing expression
executor handles the call. The JIT context and the pool must both remain open
until it returns.

Arrays backed by a confined `NJ.scope()` cannot be handed to worker threads.
When any input or output is confined to the caller, the operation falls back
on that caller, without copying or changing the existing ownership rules.
Automatic memory and arrays in `NJ.scopeShared()` can be worker-accessible.
The caller must keep all scopes open until the call returns. Concurrently
closing a shared scope can make a worker fail and can leave output partial;
all submitted workers finish before the failure reaches the caller.

`lastDecision()` reports the calling thread's latest selected strategy and
reason. `parallelRuns()` and `fallbackRuns()` count completed dispatches across
the context. A nested worker selection reports `sequential`. These diagnostics
are observational and should not be used to synchronize application logic.

## Sum and reproducibility

`pool.sum(input)` returns a scalar using the input's existing scope ownership
rules. Large C-contiguous float32/64 inputs use fixed 32,768-element chunks and
combine partials in increasing chunk order, regardless of worker count or
scheduling. Even one worker uses that same chunked algorithm for an eligible
input. The same data, dtype, algorithm and runtime therefore produce the same
result across worker counts. Floating-point grouping differs from the existing
sequential sum, so their low bits can differ. Each float32 partial and merge is
rounded to float32. The accumulation starts from positive zero, and IEEE NaN
and infinity behavior follows those additions. Small, non-C-contiguous, other
dtype and confined inputs use the existing sequential sum.

NumJVM does not change OpenBLAS thread settings and does not call BLAS from the
parallel paths above. Performance depends on array size, memory bandwidth and
provider; the [reference campaign](../benchmarks/M4-reference-protocol.md) is
still needed to calibrate thresholds and quantify scaling.
