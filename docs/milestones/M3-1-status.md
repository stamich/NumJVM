# 0.5.0-alpha implementation and reference status

The 0.5.0 branch builds on merged 0.4.0 commit
`e5f7cfd333cdde7dce84c42f309f67b77a1b9249`. The experimental candidate adds
portable real dot/vecdot and optional SIMD/OpenBLAS acceleration, typed expression
schedules, bounded scratch, an eager strategy and optional fused-vector code.
See the design records and benchmark protocol for supported semantics and limits.

## Reference campaign: open

The supplied Ryzen 9 5950X/128 GB result records a useful 0.4.0 expression run,
but its source revision and eager provider are unknown. The machine is not
available in this execution environment. The 0.5.0 three-fork reference campaign
has therefore not been run. [Protocol and commands](../benchmarks/M3-1-protocol.md)
are provided for the user's machine. Do not treat exploratory cloud measurements
as satisfying the 0.5.0 performance gates.

The reference campaign remains separate from preparation of the implementation
PR. Attach results and mark each gate passed or failed when measurements are
available. This status does not imply a release, tag or publication.

## Candidate validation

The implementation is organized as one commit per A01–A04, B01–B06,
C01–C06, D01–D06, E01–E05 and F01–F03 task. F01 provides the reproducible
campaign and marks the external Ryzen run as outstanding. B06/C06 implement
provisional selection; reference calibration and the proposed speedup budgets
remain open.

Local checks include JDK 25 builds, NumPy fixtures, Java and required SIMD
contraction tests, required LP64 OpenBLAS tests, generated-fusion correctness,
scope/concurrency tests, examples, and staged classpath/JPMS consumers. Optional
backends are tested in explicit jobs rather than inferred from discovery. Final
PR-head CI results must be checked separately; local results do not establish
remote check status or reference-machine performance.
