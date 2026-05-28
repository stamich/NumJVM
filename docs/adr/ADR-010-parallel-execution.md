# ADR-010: opt-in parallel CPU execution

Status: proposed for 0.6.0.

A reusable, explicitly closed execution context owns a bounded pool; no global
pool or implicit change to existing APIs is introduced. Calls remain synchronous.
The caller selects the worker limit; fixed, bounded chunking and deterministic
combination order do not depend on completion order. Small, unsupported or
worker-inaccessible arrays use the existing sequential implementation.

## Ownership and safety

Validate scope liveness and access on the calling thread before scheduling.
Only submit work for segments accessible to all target workers. A confined
`NJ.scope()` cannot be accessed by a pool worker: execute the original operation
on the calling thread, without copying or changing result ownership. Shared and
automatic memory may be eligible. Never assume caller accessibility implies
worker accessibility. Follow the existing output validation and overlap rules;
use parallel writes only for provably disjoint regions, with identical views
allowed only when each element is read before that element is written.

A concurrent shared-scope close may cause an FFM access failure. It is not
transactional: output may be partially written. Await every submitted worker on
success, error or interruption before returning; restore interruption status
and propagate a documented exception. No worker may access arrays after return.
Closing the context must prevent new operations, wait for active operations,
and terminate the pool. Do not hold locks in a way that deadlocks nested calls;
worker-originated nested work must run sequentially or be rejected explicitly.
The caller must not mutate inputs or close their scope concurrently unless it
accepts the documented failure/partial-output behavior.

## Numeric and native interaction

Parallel sum uses fixed logical chunks and a fixed merge order independent of
worker count and scheduling. Reproducibility is scoped to the same algorithm,
dtype, backend and Java environment; it does not promise identical results to
sequential accumulation. Document NaNs, infinities and signed zero according to
the implemented arithmetic. Do not launch BLAS from NumJVM worker tasks or
modify process-wide native thread settings. Performance thresholds are
provisional until the reference campaign. No multiworker speedup is guaranteed.
