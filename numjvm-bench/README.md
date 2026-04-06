# NumJVM benchmarks

JMH benchmarks. The M0 set measures the overhead the architecture adds (see [`docs/benchmarks/M0.md`](../docs/benchmarks/M0.md));
the M1 set measures ufuncs and reductions against NumPy (see [`docs/benchmarks/M1.md`](../docs/benchmarks/M1.md)).

| Class                   | Benchmarks (M0 specification name)                                 | Parameters   |
|-------------------------|--------------------------------------------------------------------|--------------|
| `AllocationBenchmark`   | `arrayAllocation` (ArrayAllocation), `heapVsNative` (HeapVsNative) | size, memory |
| `ViewBenchmark`         | `viewCreation`, `reshape`, `transpose`, `broadcastCreation`        | size         |
| `MemoryAccessBenchmark` | `contiguousRead`, `stridedRead`                                    | size, layout |
| `KernelBenchmark`       | `miniAdd`, `miniSum`                                               | size, layout |
| `ScopeBenchmark`        | `scopeAllocation`, `scopeClose`                                    | size, mode   |

M1 (ufuncs and reductions; every call allocates its result unless noted):

| Class                     | Benchmarks                                                                                              | Parameters     |
|---------------------------|---------------------------------------------------------------------------------------------------------|----------------|
| `UfuncBenchmark`          | `add`, `addScalar`, `addRow`, `addOut` (into a preallocated array), `multiply`, `divide`, `maximum`, `less`, `sqrt`, `exp`, `sin` | size, layout   |
| `UfuncDTypeBenchmark`     | `add`, `multiply`, `divide`, `sqrt`, `exp`                                                              | dtype          |
| `ReductionBenchmark`      | `sumAll`, `sumAxis0`, `sumAxis1`, `maxAll`, `meanAll`, `prodAxis0`                                       | size, layout   |
| `ReductionDTypeBenchmark` | `sum`, `mean`, `max`, `any`                                                                             | dtype          |

The M1 benchmarks use 1,024, 65,536 and 1,048,576 elements (the per-dtype ones only the largest) holding
`0.5 + 4i/n`, so that every function is defined on them. Their NumPy counterparts have the same names:

Sizes are 16, 128, 1,024, 65,536, 1,048,576 and 16,777,216 float64 elements, arranged as a near-square matrix. Layouts
are `C`, `F`, `TRANSPOSED` (a transposed view), `STRIDED`
(every other column) and `BROADCASTED` (one row broadcast to all rows).

```bash
./gradlew :numjvm-bench:jmh                                   # everything: several hours
./gradlew :numjvm-bench:jmh -PjmhArgs="KernelBenchmark -p size=65536 -prof gc"
./gradlew :numjvm-bench:jmh -PjmhArgs="-h"                    # all JMH options
```

```bash
./gradlew :numjvm-bench:jmh -PjmhArgs="UfuncBenchmark|UfuncDTypeBenchmark|ReductionBenchmark|ReductionDTypeBenchmark -f 3 -rf csv -rff jmh-m1.csv"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numjvm-bench/numpy/bench_numpy.py --suite m1 > numjvm-bench/numpy-m1.csv
python3 numjvm-bench/numpy/compare.py numjvm-bench/jmh-m1.csv numjvm-bench/numpy-m1.csv   # NumJVM / NumPy per row
```

Always pass `-prof gc` when checking allocation: element access and iteration must show 0 B/op, and a ufunc only the
bytes of its result.

M2 adds `M2Benchmark` (mixed conversion and integer reduction allocation/out pairs)
and `MatmulBenchmark` (square/rectangular/vector-sized shapes, automatic/explicit
native memory and packing). Use `-PsimdBench` to include the optional provider.
Set `numjvm.linalg` and `numjvm.openblas.path` through JVM options inherited by
forks, and record those settings. Example smoke command:

```sh
./gradlew :numjvm-bench:jmh -PjmhArgs='M2Benchmark.mixedAddOut -p size=65536 -f 3 -wi 1 -w 100ms -i 1 -r 100ms -prof gc'
./gradlew :numjvm-bench:nativeAccounting
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 numjvm-bench/numpy/bench_m2.py
```

`nativeAccounting` samples counters outside timing loops; GC profiling alone cannot
measure native scratch. Short smoke runs verify instrumentation and dispatch, not
performance targets. Compare matrix packing-inclusive and already-native layouts
separately. Automatic native inputs avoid retaining every allocating result in a trial-long scope.

Expression lifecycle and native accounting are measured separately from the warm
throughput benchmark:

```sh
./gradlew :numjvm-bench:jmh -PjmhArgs='ExpressionLifecycleBenchmark -f 3 -wi 3 -i 5 -prof gc'
./gradlew :numjvm-bench:expressionCompileProbe
./gradlew :numjvm-bench:expressionMemoryProbe
```

The first-execution benchmark includes compilation and invocation. The diagnostic
compile probe records their components in a separate single process; use JMH forks
for reproducible timing. Native counters include tracked scratch and scoped array
storage; peak is process-wide and cannot be interpreted as per-operation peak.
Heap allocation is reported by the JMH GC profiler, not by native counters.
