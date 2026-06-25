# 0.6.0 fusion performance status

The original reference measurements remain documented in
[0.5.0 results](M3-1-reference-results.md). They preceded the generated-loop
changes and cannot establish the performance of this revision. The user's
Ryzen 9 5950X reference machine is not available in this environment.

A one-fork JDK 25 shared-machine smoke comparison of C-order float32 ADD and
CHAIN with 1,048,576 elements found substantially lower AUTO allocation after
generating the whole vector traversal loop; see the
[investigation](M4-fusion-investigation.md). Float64, mixed, C and F smoke
runs also completed. These short, separate runs can identify gross regressions,
but do not satisfy three-fork reference gates or establish stable speedups.
AUTO now selects eager for direct single-ufunc expressions with a SIMD provider;
it must be evaluated as selected, not relabeled as a fused ADD measurement.

| Gate | Status |
|---|---|
| 1.5× large fused-output target | Open: rerun eligible multi-operation cases on the reference host. |
| 1.8× allocating regression guard | Open: rerun the specified allocating cases on the reference host. |
| Small and medium AUTO budgets | Open: the implementation changed after the old reference measurements. |
| Real dot/vecdot performance | Open: run the new contraction benchmark with backend provenance. |
| 0.6.0 worker scaling | Open: requires the parallel implementation and reference campaign. |

Do not lower or waive the earlier targets. Record clean revision, patch status,
JDK, SIMD/native provider and thread settings with raw JSON and logs.

The [0.6.0 scaling protocol](M4-reference-protocol.md) is ready, but the
reference campaign is outstanding. Crossover thresholds and multiworker
speedups are not established. A short shared-machine benchmark verified the
harness executes; it does not provide reference performance evidence.
