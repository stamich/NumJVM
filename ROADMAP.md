# NumJVM roadmap

| Milestone | Content                                                                             | Status                                                |
|-----------|-------------------------------------------------------------------------------------|-------------------------------------------------------|
| M-1       | Feasibility spike: allocation, iteration, kernels, ADR-001–003                      | Done (Nov 2025)                                       |
| **M0**    | Core array and memory foundation, `.npy`, NumPy oracle — **0.1.0-alpha**            | Done (Nov 2025), see [summary](docs/milestones/M0.md) |
| M1        | Elementwise engine: iteration plans, kernel registry, ufuncs, reductions, promotion | Next                                                  |
| M2        | Fast paths: Vector API, BLAS/LAPACK through FFM, multithreading                     | Planned                                               |
| M3        | JIT and fusion: tracing, IR, ClassFile API code generation                          | Planned                                               |
| M4        | Ecosystem: Kotlin and Scala APIs, Arrow, DLPack, notebooks, Maven Central           | Planned                                               |
| M5        | Advanced computing: GPU and experimental backends                                   | Planned                                               |
| M6        | Production readiness: API freeze, compatibility, 1.0                                | Planned                                               |

## M1 — Elementwise engine (re-estimated after M0)

Goal: NumPy-level arithmetic coverage for real dtypes with correct promotion, and transposed and broadcast operands
within 2× of contiguous ones.

The original estimate was 6–8 weeks. M0 took 25 calendar days for 37 planned ideal days (about 1.5 ideal days per
calendar day of part-time work), and its phases with an external oracle ran ahead of plan. The work below is 34 ideal
days, which at the same rate and with 25 % buffer is **about 4–5 calendar weeks**.

| ID        | Task                                                                                                      | Est. (ideal days) |
|-----------|-----------------------------------------------------------------------------------------------------------|------------------:|
| M1.1      | Iteration plan: dimension coalescing, loop reordering by stride, contiguous fast path selection (ADR-004) |                 4 |
| M1.2      | Kernel registry keyed by operation and dtype loop signature; scalar Java loops generated from templates   |                 4 |
| M1.3      | Type promotion per the pinned Array API revision; `result_type`, `can_cast`                               |                 3 |
| M1.4      | Binary ufuncs: add, subtract, multiply, divide, floor_divide, remainder, power, minimum, maximum          |                 4 |
| M1.5      | Comparison and logical ufuncs returning bool                                                              |                 2 |
| M1.6      | Unary ufuncs: negative, abs, sqrt, exp, log, sin, cos, tanh, floor, ceil, round                           |                 3 |
| M1.7      | Reductions over axes with `keepdims`: sum, prod, min, max, mean, any, all; pairwise for floats            |                 5 |
| M1.8      | `out=` arrays and allocator-aware results (scopes); in-place operations                                   |                 2 |
| M1.9      | Complex arithmetic for complex64/128 (ends storage-only status)                                           |                 3 |
| M1.10     | Oracle extension: generated cases per ufunc × dtype × layout, Hypothesis-driven                           |                 2 |
| M1.11     | Benchmarks: transposed and broadcast within 2× of contiguous; full matrix on dedicated hardware           |                 2 |
| **Total** |                                                                                                           |            **34** |

Exit criteria: oracle green for all ufuncs over all real dtypes and five layouts; summing a transposed array within 2×
of a contiguous one; `ValidationKernels` removed in favour of the public API.
