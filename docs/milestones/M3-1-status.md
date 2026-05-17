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
