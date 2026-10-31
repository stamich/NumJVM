# ADR-007: Explicit expression execution

Status: proposed for M3

## Context

Eager operations allocate intermediate arrays. An explicit expression graph can
reuse a plan and limit intermediate storage while leaving eager APIs unchanged.

## Decision

The experimental JIT module will trace symbolic input slots once and evaluate
them synchronously against invocation arguments. It will initially support
float32 and float64 arithmetic (add, subtract, multiply, divide, negative,
absolute, square and square root) and weak double constants. Resolution and
rounding occur at each node, matching the existing eager scalar operations.

Expression planning validates every intermediate shape and infers its eager
allocation order without allocating intermediate arrays. Allocating results
use the first scoped argument's allocator, otherwise automatic storage.

The fallback executes blocks through existing kernels using bounded scratch;
eligible runs may instead use kernels generated with the Java Class-File API.
An identical input/output view is eligible for direct execution when all
overlapping inputs are safe and input values are read before output is written.
Other hazardous overlap requires an independent result before copyback. Output
validation precedes writes. Plans and caches must not retain invocation arrays
or their owners. Cache entries and graph size are bounded.

## Scope and sequence

Reductions, comparisons, conditional selection, general transcendental
functions, fused SIMD, parallel workers and LAPACK are outside M3. The proposed
sequence is M3/0.4.0 for expression fusion, 0.5.0 for linalg extensions, 0.6.0
for parallel CPU execution, and subsequently M4 ecosystem integration. Exact
later feature scopes require separate proposals.

## Consequences

The generated and block engines must be tested against eager results, including
float32 intermediates, signed zero, broadcast layouts, scopes and aliasing.
Benchmarking must compare eager SIMD, block and generated paths, including cold
compilation and scratch costs. This ADR records a proposed contract; it does
not claim that the engines have been implemented.
