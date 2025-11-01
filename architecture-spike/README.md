# NumJVM architecture spike (M-1)

Throwaway code that answers the questions the M0 API depends on. Nothing here is
part of the library and nothing here is published. The conclusions are recorded in
[`results.md`](results.md) and in the ADRs under [`../docs/adr`](../docs/adr).

| Step | Question | Code |
| --- | --- | --- |
| S1 | Heap vs native allocation cost, heap segment alignment | `MemorySpike` |
| S2 | Minimal float64 array with byte strides and an allocation-free iterator | `SpikeArray`, `Odometer`, `SpikeArrayCheck` |
| S3 | Strided `add` with broadcasting, pairwise `sum`, fast path vs generic loop | `StridedAdd`, `SumReduction`, `KernelSpike` |
| S4 | Lifetime of 100k short-lived off-heap arrays under `Arena.ofAuto()` | `LifetimeSoak` |
| S5 | JMH harness and NumPy baseline for the same operations | `benchmarks/` |

## Running

Requires JDK 25. No build tool is needed:

```bash
cd architecture-spike
javac -d out src/spike/*.java
java -cp out spike.MemorySpike            # S1
java -cp out spike.SpikeArrayCheck        # S2
java -cp out spike.KernelSpike            # S3
java -Xmx512m -cp out spike.LifetimeSoak  # S4
```

The JMH benchmarks (S5) are described in [`benchmarks/README.md`](benchmarks/README.md).
