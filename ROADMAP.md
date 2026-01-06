# NumJVM roadmap

| Milestone | Content                                                                             | Status                                                |
|-----------|-------------------------------------------------------------------------------------|-------------------------------------------------------|
| M-1       | Feasibility spike: allocation, iteration, kernels, ADR-001–003                      | Done (Nov 2025)                                       |
| **M0**    | Core array and memory foundation, `.npy`, NumPy oracle — **0.1.0-alpha**            | Done (Nov 2025), see [summary](docs/milestones/M0.md) |
| **M1**    | Elementwise engine: iteration plans, kernel registry, ufuncs, reductions, promotion — **0.2.0-alpha** | Done (Jan 2026), see [summary](docs/milestones/M1.md) |
| M2        | Fast paths: Vector API, BLAS/LAPACK through FFM, multithreading                     | Next                                                  |
| M3        | JIT and fusion: tracing, IR, ClassFile API code generation                          | Planned                                               |
| M4        | Ecosystem: Kotlin and Scala APIs, Arrow, DLPack, notebooks, Maven Central           | Planned                                               |
| M5        | Advanced computing: GPU and experimental backends                                   | Planned                                               |
| M6        | Production readiness: API freeze, compatibility, 1.0                                | Planned                                               |

## M2 — Fast paths (re-planned after M1)

Goal: close the gap to NumPy on contiguous data, where M1's scalar loops lose to NumPy's SIMD loops, and add the first
linear algebra. M1 left the architecture ready: loops already receive a contiguous path per run, native buffers are
64-byte aligned, and the oracle with its ULP contract checks any faster implementation against NumPy unchanged.

M1 took 40 calendar days for 41 planned ideal days. At the same pace, with a 20 % buffer for the Vector API's
incubator status, the 37 ideal days below take **about 6–7 calendar weeks**.

| ID        | Task                                                                                                          | Est. (ideal days) |
|-----------|---------------------------------------------------------------------------------------------------------------|------------------:|
| M2.1      | ADR-006: Vector API adoption (incubator flag, fallback to scalar loops, per-dtype species, alignment)         |                 1 |
| M2.2      | Vector templates in the code generator: contiguous and broadcast paths of arithmetic, comparisons, extrema    |                 5 |
| M2.3      | Vectorized reductions: pairwise sum with vector accumulators, bit-identical to NumPy's blocking               |                 3 |
| M2.4      | Vectorized `sqrt`, `abs`, rounding; transcendental functions stay scalar unless within the D6 contract         |                 2 |
| M2.5      | Allocation: pooled scratch for operand conversions, `out=` everywhere internally, `NJ.zeros` without double zeroing |           3 |
| M2.6      | Multithreading: split large runs over a fork-join pool above a size threshold; deterministic reductions       |                 4 |
| M2.7      | `matmul`/`dot` for float32/float64: blocked Java kernel, then BLAS `dgemm`/`sgemm` through FFM downcalls       |                 5 |
| M2.8      | BLAS discovery and packaging (system OpenBLAS, opt-out, error messages); oracle cases for matmul              |                 3 |
| M2.9      | LAPACK through FFM: `solve`, `inv`, `det`, `eigh`, `svd` for float64                                          |                 5 |
| M2.10     | Complex ordering, `power` and trigonometric loops (the gaps listed in NOT_IN_NUMJVM)                          |                 2 |
| M2.11     | Benchmarks against NumPy with SIMD on and off; dedicated or large hosted runner, three forks                   |                 2 |
| M2.12     | Docs, retrospective, release 0.3.0-alpha                                                                      |                 2 |
| **Total** |                                                                                                               |            **37** |

Exit criteria: contiguous float64 `add`, `multiply` and `sum` within 1.5× of NumPy at 1M elements; `matmul` of two
1024 × 1024 float64 matrices within 1.2× of NumPy with the same BLAS; no oracle regression; the scalar fallback still
passes the full oracle with the Vector API disabled.
