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
