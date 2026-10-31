# M3 draft candidate status

Starting point: `d45ef0115b2ee67713a08b8edd3cd369078df946`, Java 25.
The draft adds an optional JIT module with explicit expression graphs, block
execution and generated scalar kernels. Its API is experimental. Refer to the
[expression guide](../design/expression-execution.md) for supported operations
and to the [benchmark report](../benchmarks/M3.md) for exploratory measurements.

The M2 reference-machine performance campaign remains open. A dedicated M3
reference machine has not been designated; small-array budgets, SIMD comparison,
compiler cold costs and the target speedup remain unverified. The shared-machine
sample did not demonstrate the proposed speedup. Remote checks, required-check
settings and publication on the candidate commit require verification. Do not
interpret this draft as a tag, package release, or completed performance gate.
