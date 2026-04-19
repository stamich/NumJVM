# Experimental expression execution

The optional `numjvm-jit` artifact records explicit symbolic expressions. A trace
callback runs once to build a graph; evaluation uses new invocation arrays and
never calls the builder. `Jit.context()` selects automatic execution. Use
`Jit.context(Jit.Mode.BLOCK)` or `Jit.Mode.REQUIRED` to force the block executor
or require generated execution. A context must remain open while starting an
operation; `close()` clears its bounded kernel cache.

```java
try (var context = Jit.context()) {
    var expression = context.trace(2, p -> p.input(0).multiply(p.input(1)).add(2).sqrt());
    var result = expression.eval(a, b);
    expression.evalInto(out, a, b);
}
```

This alpha supports float32/float64 inputs, weak double constants, addition,
subtraction, multiplication, division, negative, absolute, square, and sqrt.
It does not yet support comparisons, selection, general transcendental functions,
reductions, or matrix multiplication inside a graph. Evaluate first and then
call existing eager APIs for those operations.

Each node resolves dtype and broadcasting separately. Float32 intermediates are
rounded before later float64 operations. Result layout follows the eager
allocation-order rules. Allocating results use the first scoped input, otherwise
automatic storage. Inputs and outputs must be accessible on the calling thread.
A compatible identical-view output can be used in-place; hazardous overlap is
computed through a temporary. Output shape and dtype must match exactly.

Block mode uses bounded scratch and existing ufunc kernels. Required mode demands
an eligible generated run and rejects unsupported layouts or large graphs before
writing. Automatic mode considers size, graph complexity, dtype, provider,
generated eligibility, cached kernels and the eager temporary budget. Its
thresholds are provisional; they are not a performance guarantee. Context
statistics distinguish block and generated runs, compilations, cache hits and
evictions; `eagerRuns()` counts whole-array runs. `lastDecision()` reports the
current thread's latest selection and reason, and `clearThreadDecision()` removes
it. The cache holds no invocation arrays.

`fusedVectorProvider()` reports service discovery only; it does not prove that a
particular graph is supported or that vector instructions executed. The provider
is optional and the JIT remains usable without the incubating Vector API.

Run the example with `./gradlew :examples:fused-expressions:run`. Performance
validation and unresolved gates are described in [the benchmark protocol](../benchmarks/M3-protocol.md).

## Block scratch configuration

An experimental context may set the operation-local block scratch byte cap with
`Jit.context(mode, cacheEntries, scratchBytes)`, from 2,048 to 65,536 bytes.
The default is 65,536 bytes. This cap excludes a full independent result used
to protect hazardous output overlap. The existing process-wide scratch limit also
applies. `ExpressionScratchBenchmark` compares several caps; the default has not
yet been tuned on the designated reference machine.

`Jit.Mode.EAGER` explicitly executes whole-array ufuncs and can allocate full-sized
intermediates. Its default conservative intermediate budget is 16 MiB. Configure
both budgets with `Jit.context(mode, cacheEntries, scratchBytes, eagerBytes)`.
The eager budget excludes the result, alias-protection result and the existing
bounded ufunc conversion scratch; a graph over budget is rejected before writes.
