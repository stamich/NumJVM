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

`ExpressionScratchBenchmark` was smoke-tested at 4,096, 16,384 and 65,536 byte
caps with one short fork. That cold, shared-host run does not establish a best
cap; keep the default provisional until a reference-machine comparison.

A further single-fork, three-warmup/three-measurement check of generated-out at
65,536 float64 values compared `e2f6f83` with the working-tree loop-hoisting change.
Both reported approximately 295 µs/op; the changed run varied from 172 to 523
µs/op. Raw [before](m3-1-exploratory/b02-before.json) and
[after](m3-1-exploratory/b02-after.json) measurements do not establish a speedup.
