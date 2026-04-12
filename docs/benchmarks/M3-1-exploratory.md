# M3.1 exploratory block execution

On a shared cloud worker, one JMH fork with three one-second warmups and three
one-second measurements tested `ExpressionBenchmark.blockOut`, 65,536 float64
values. Before the direct input/output change, commit `0aeb977` measured 325.1
µs/op; the working-tree implementation measured 236.0 µs/op. Raw results:
[before](m3-1-exploratory/b03-before.json), [after](m3-1-exploratory/b03-after.json).

This was a single fork on shared hardware, measured sequentially after then
before. It checks whether the optimization is promising; it is not a reference
campaign, a controlled causal claim, or proof of the 0.5.0 performance gates.
The implementation passes contiguous input runs directly to kernels and writes
eligible root operations directly to output, while keeping gathers/scatters for
other layouts and independent output storage for hazardous overlap.
