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
