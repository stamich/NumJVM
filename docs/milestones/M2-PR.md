# Implement 0.3.0-alpha CPU backends and portable matrix multiplication

M1 materializes whole-array operand conversions and reduction results, and has no
portable matrix multiplication. This candidate bounds binary conversion scratch,
adds mixed input/accumulator reductions and direct output paths, and introduces
optional SIMD plus float32/64 matmul with complete Java fallback and checked LP64
OpenBLAS GEMM.

Core remains usable without an incubator module or native library. Matrix outputs
preserve scope ownership, arbitrary valid input strides and alias-safe copyback.
The API, native baseline and acceleration subset are documented with runnable examples.

The branch has one commit for each task A01–F04 (48 total), using the maintainer's
requested author/committer identity and timestamps in Europe/Warsaw. Those historical
Git dates are not measurement dates; validation occurred on 30 October 2026.
No co-author trailers are added.

Validation on the candidate sources:

- Full build and existing oracle pass in scalar-only, installed-but-disabled, and
  required-vector core/I/O modes; vector tests additionally assert actual execution.
- Java, forced OpenBLAS and auto linalg contracts pass, including 24 NumPy cases,
  high-precision bounds, identity properties, aliases, layouts and lifecycle errors.
- Final report: 6,400 passing test executions, seven context-specific skips, zero
  failures/errors. Line coverage: core 98.43%, I/O 94.41%, vector 97.25%, linalg 93.89%.
- Classpath/JPMS Maven Local consumers pass with/without SIMD, including unresolved
  classpath-provider fallback. Source/Javadoc artifacts are built and published locally.
- Existing fixtures and matrix fixtures reproduce; NumPy reads 4,245 Java-produced
  files with zero failures. Examples execute in all advertised modes.
- Three-fork matrix measurements and raw uncertainty are retained. Native accounting
  measures zero retained scratch bytes after 100 mixed output calls.

Keep this PR draft pending the reference-machine performance campaign and remote
CI/Sonar review. Shared cloud measurements tune provisional packing-aware dispatch;
they do not establish the 1M-element or same-BLAS 1024² release targets. No correctness
or memory gate is waived. No release tag or package publication is performed.

Details: [M2 status](M2-status.md), [release checklist](M2-release-checklist.md),
[validation](M2-validation.json), and [benchmark report](../benchmarks/M2.md).
