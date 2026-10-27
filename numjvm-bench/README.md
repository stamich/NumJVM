# NumJVM benchmarks

JMH benchmarks for M0. They measure the overhead the architecture adds, not peak arithmetic throughput; see [
`docs/benchmarks/M0.md`](../docs/benchmarks/M0.md) for results and methodology.

| Class                   | Benchmarks (M0 specification name)                                 | Parameters   |
|-------------------------|--------------------------------------------------------------------|--------------|
| `AllocationBenchmark`   | `arrayAllocation` (ArrayAllocation), `heapVsNative` (HeapVsNative) | size, memory |
| `ViewBenchmark`         | `viewCreation`, `reshape`, `transpose`, `broadcastCreation`        | size         |
| `MemoryAccessBenchmark` | `contiguousRead`, `stridedRead`                                    | size, layout |
| `KernelBenchmark`       | `miniAdd`, `miniSum`                                               | size, layout |
| `ScopeBenchmark`        | `scopeAllocation`, `scopeClose`                                    | size, mode   |

Sizes are 16, 128, 1,024, 65,536, 1,048,576 and 16,777,216 float64 elements, arranged as a near-square matrix. Layouts
are `C`, `F`, `TRANSPOSED` (a transposed view), `STRIDED`
(every other column) and `BROADCASTED` (one row broadcast to all rows).

```bash
./gradlew :numjvm-bench:jmh                                   # everything: several hours
./gradlew :numjvm-bench:jmh -PjmhArgs="KernelBenchmark -p size=65536 -prof gc"
./gradlew :numjvm-bench:jmh -PjmhArgs="-h"                    # all JMH options
```

Always pass `-prof gc` when checking allocation: element access and iteration must show 0 B/op.
