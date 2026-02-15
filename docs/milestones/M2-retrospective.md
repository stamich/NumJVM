# M2 implementation retrospective

This is an implementation candidate for 0.3.0-alpha, not a published release.
The [status](M2-status.md) and [benchmark report](../benchmarks/M2.md) identify the
remaining reference-performance and remote quality gates.

## Delivered checkpoints

1. Scalar execution: bounded binary conversion leases, mixed input/accumulator
   integer reductions, virtual cast-layout integer mean and compatible direct outputs.
2. Optional SIMD: generated kernels and scalar fallback, exact pairwise tree,
   direct parity/dispatch tests, classpath and JPMS consumer coverage.
3. Multiplication: complete portable real matmul with synchronous batches,
   checked OpenBLAS LP64 bindings, packing, alias copyback and ownership checks.

The three-fork matrix experiment and its raw uncertainty are retained. Allocation
and output paths are distinct benchmarks. Native accounting shows deterministic
scratch release; it does not claim zero allocations. Small rectangular, batched,
additional-platform and designated-reference campaigns remain future evidence.

## Findings that changed implementation

- Short JMH samples were too noisy to select thresholds. Longer sequential runs
  showed packing cost erases the 32³ heap-backed advantage. Auto selection now
  distinguishes native-compatible storage from packing paths.
- A service jar being discoverable does not prove its incubator dependency is
  resolved. Force linkage during provider construction so auto mode can explain
  unavailability before selecting a numerical loop.
- Removing a strided integer cast can change reduction grouping. A virtual cast
  layout preserves the reference tree without materializing converted input.
- Direct output initialization must follow lifecycle validation, not precede it.
- Float32 BLAS and sequential Java contractions can differ while both meet the
  matrix error contract. Exact Java-order tests must invoke Java directly, while
  cross-backend tests need the stated bounds and high-precision reference.
- Java-only coverage cannot exercise a native downcall. Keep native source in
  coverage and enforce combined coverage through a required-native integration job.
- Native input lifetimes must extend through counter snapshots as well as calls;
  the diagnostic probe uses reachability fences to avoid confusing Cleaner release
  of trial inputs with scratch cleanup.

## Delivery record and follow-ups

The 48 task IDs are commit boundaries, not a claim of 48 completed release gates.
Maintainer-requested author/committer timestamps span January–February 2026.
Actual validation occurred on 30 October 2026. No reliable actual engineering-day
measurement was collected; do not infer it from those Git dates or compare it to
74 ideal days as measured effort.

Before release: designate reference hardware, finalize regression budgets, run the
full M1/candidate campaign and inspect remote CI/Sonar evidence. The existing
performance objectives are not waived. Native platforms beyond Linux x86-64 remain
unsupported. Worker pools, factorizations and additional numerical APIs require
separate ownership/accuracy/versioning decisions before M3.
