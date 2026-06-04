# Fused vector investigation

The supplied JDK 25 recordings reproduce the 0.5.0 allocation issue on the
reference host: float32, 1,048,576 elements, JMH GC profiler: AUTO ADD
16,785,621 B/op and CHAIN 33,568,252 B/op versus eager approximately 1,089
and 3,267 B/op. JFR allocation samples point to `FloatFusedKernel.run` vector
loads, including `float[]` and `Float256Vector` objects. Samples are not counts;
see [reference results](M3-1-reference-results.md) for provenance and limits.

The loop loads vectors into a `FloatVector[]` and sends that array through the
`FloatExpression.evaluate` interface at every vector-sized iteration. This
can cause objects to escape scalar replacement, but the profile does not prove
why HotSpot allocated them. Generate or otherwise specialize a whole traversal
loop and measure both allocation and speed before concluding that a change
fixes the cause. Confirm float64 and mixed independently. Small AUTO overhead
occurs even when eager is selected and needs a separate investigation.

`FusedAllocationBenchmark` provides a focused JMH regression matrix. Run with
`-prof gc` and inspect `gc.alloc.rate.norm` as well as elapsed time, recording
provider and `NUMJVM_AUTO_DECISION` from its setup. For example, pass
`-PjmhArgs='FusedAllocationBenchmark -p size=1048576 -p layout=C -p types=FLOAT32 -p chain=ADD,CHAIN -prof gc'`
to `:numjvm-bench:jmh` with `-PsimdBench` and the required vector JVM options.

An exploratory shared-machine JDK 25 run (one fork, three one-second warmups
and measurements, 1,048,576 float32 C-order elements) compared the old array
based path with a generated whole-run loop. AUTO ADD allocation fell from about
12.6 MB/op to 8.1 kB/op and CHAIN from 25.2 MB/op to 13.7 kB/op. These are
separate runs on a shared machine, not reference-machine acceptance results.
The generated loop keeps vector intermediates in locals, preserving a separate
scalar-tail path; mixed graphs still use the original vector-array path until
independently improved and measured. Repeat the full matrix on the reference
machine before closing performance gates.
