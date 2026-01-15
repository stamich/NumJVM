# M2 allocation investigation

Measurements below are three-fork **smoke** observations, not stable performance evidence.

| Elements | Result bytes | addOut GC bytes/op |
| --- | --- | --- |
| 1024 | 8192 | 1152.7 |
| 65536 | 524288 | 1321.9 |
| 1048576 | 8388608 | 4050.2 |

The 65,536-element float64 result is 524,288 bytes: below the default
1,048,576-byte heap threshold. Native-only allocator changes cannot remove
that result allocation. `NativeAllocator.allocate` delegates to Arena.allocate
and contains no extra explicit clear. Keep current initialization guarantees.

Decision: first remove full operand conversion arrays and temporary reduction
results. Use per-operation native scratch with deterministic release and a global
active-byte budget. Retain no pooled buffers initially. Benchmark pooling separately
before accepting lifecycle complexity. Native accounting must include scratch.

No allocator threshold or zero-fill behavior is changed by this investigation.

## Scalar checkpoint and layout decision

The full scalar/oracle test run passes after bounded conversion, widening integer
reductions, contiguous integer mean and direct reduction output integration.
The repeated three-fork runner smoke is retained in `m2-scalar-candidate/`.
Its short samples do not justify changing heap/native thresholds or claiming a
speedup. Reject threshold/zero-fill changes and extra layout dispatch for now.

Direct output is currently eligible for exact accumulator dtype, C-contiguous,
non-overlapping outputs. Strided/casting/overlapping outputs preserve temporary
fallback. Integer mean fuses conversions for C/F-contiguous inputs; non-contiguous
inputs preserve cast-then-reduce grouping. Extending that path needs a virtual
cast-layout traversal, not a naive reassociation of input reads.

Conversion scratch is 64 KiB per active mixed ufunc, no retention after a call,
and at most 16 MiB process-wide including nested/concurrent leases. Exhaustion
raises MemoryException before acquiring a new lease; no caller blocks holding
another lease. Arena bytes appear in NJ.memoryStats(). No pooling optimization
is accepted without full measurements demonstrating a benefit.
