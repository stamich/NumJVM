# 0.5.0-alpha implementation and reference status

The experimental candidate adds portable real dot/vecdot and optional
SIMD/OpenBLAS acceleration, typed expression schedules, bounded scratch, an
eager strategy and optional fused-vector code. It builds on merged 0.4.0
commit `e5f7cfd333cdde7dce84c42f309f67b77a1b9249`.

## Reference results: performance issues and open gates

The user supplied full scalar and SIMD Ryzen 9 5950X campaigns and follow-up
measurements and JFR profiles. The first campaigns identify commit `4aff717`
but report a dirty checkout. Follow-up C-order measurements started from a
clean checkout of that revision. They reproduce substantial AUTO regressions:
for large float32 ADD and CHAIN, fused-vector AUTO was approximately 19.7×
and 9.3× slower than eager and allocated approximately 16 and 32 MiB per
operation. Correcting the F-order benchmark intermediates also reversed the
previous apparent F-order advantage in the tested cases.

JFR samples implicate vector-load allocations in the fused path; their precise
compiler cause remains a hypothesis. The representative large fused-output
cases do not meet the proposed 1.5× speedup. The full allocating, AUTO-budget,
contraction and remaining reference gates are not established. Details,
provenance and limitations are in the [results report](../benchmarks/M3-1-reference-results.md);
[protocol and commands](../benchmarks/M3-1-protocol.md) support follow-up runs.
Do not infer release readiness or publication from this status.

## Candidate validation

The implementation is organized as one commit per A01–A04, B01–B06,
C01–C06, D01–D06, E01–E05 and F01–F03 task. AUTO and native selection
thresholds remain provisional pending reference calibration. Local checks
included JDK 25 builds, NumPy fixtures, Java and required SIMD contraction
tests, required LP64 OpenBLAS tests, generated-fusion correctness,
scope/concurrency tests, examples, and staged classpath/JPMS consumers. Check
CI at the actual PR head separately; earlier results do not establish the
status of later commits.
