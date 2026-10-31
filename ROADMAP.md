# NumJVM roadmap

| Milestone | Content                                                                             | Status                                                |
|-----------|-------------------------------------------------------------------------------------|-------------------------------------------------------|
| M-1       | Feasibility spike: allocation, iteration, kernels, ADR-001–003                      | Done (Nov 2025)                                       |
| **M0**    | Core array and memory foundation, `.npy`, NumPy oracle — **0.1.0-alpha**            | Done (Nov 2025), see [summary](docs/milestones/M0.md) |
| **M1**    | Elementwise engine: iteration plans, kernel registry, ufuncs, reductions, promotion — **0.2.0-alpha** | Done (Jan 2026), see [summary](docs/milestones/M1.md) |
| M2 | Optional CPU acceleration and portable matmul — **0.3.0-alpha** | Implementation candidate; release evidence pending |
| M3        | JIT and fusion: tracing, IR, ClassFile API code generation — **0.4.0-alpha** | Merged; reference-performance evidence pending |
| M3.1      | Fusion performance and real dot/vecdot — **0.5.0-alpha** | Implementation candidate; reference-performance evidence pending |
| M4        | Ecosystem: Kotlin and Scala APIs, Arrow, DLPack, notebooks, Maven Central           | Planned                                               |
| M5        | Advanced computing: GPU and experimental backends                                   | Planned                                               |
| M6        | Production readiness: API freeze, compatibility, 1.0                                | Planned                                               |

## M2 — accelerated CPU execution and portable matmul

The [48-task implementation plan](docs/milestones/M2-plan.md) replaces the previous
37-day outline. Its estimate is 74 ideal engineering days plus contingency; historical
Git timestamps do not measure actual effort or elapsed duration.

The implementation candidate adds bounded conversions, widening reductions,
optional SIMD and float32/64 matmul with Java fallback plus optional OpenBLAS LP64.
See [actual status and remaining gates](docs/milestones/M2-status.md),
[CPU architecture](docs/design/cpu-backends.md), and [measurements](docs/benchmarks/M2.md).
The performance reference campaign and remote CI/quality evidence remain open.

Thread pools, LAPACK factorizations, dot/vecdot, complex matmul and additional native
providers are explicit follow-ups. Decide their versioning and sequence before
starting M3; completion of this candidate does not imply those features shipped.
No multiworker scaling target applies to M2.

## Proposed post-M3 sequence

0.5.0 adds real dot/vecdot and fusion improvements; see its
[status and remaining gates](docs/milestones/M3-1-status.md). Plan 0.6.0 for
worker pools and deterministic parallel reductions after an ownership design.
Consider a separately scoped 0.7.0 for portable solve/Cholesky and optional LAPACK
before M4 ecosystem integration. Outstanding M2 and M3 reference-performance
evidence remains open; no publication is implied by merging implementation code.
