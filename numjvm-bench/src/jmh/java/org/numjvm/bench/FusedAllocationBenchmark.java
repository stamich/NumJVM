package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.*;

/** Use JMH's gc profiler to compare warm fused and eager output allocation. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class FusedAllocationBenchmark {
    @Param({"65536", "1048576"}) public int size;
    @Param({"FLOAT32", "FLOAT64", "MIXED"}) public String types;
    @Param({"ADD", "CHAIN"}) public String chain;
    @Param({"C", "F"}) public Layouts.Layout layout;
    private ExpressionMatrixBenchmark matrix;

    @Setup public void setup() {
        matrix = new ExpressionMatrixBenchmark();
        matrix.size = size; matrix.types = types; matrix.chain = chain; matrix.layout = layout;
        matrix.setup();
    }
    @Benchmark public NDArray autoOut() { return matrix.autoOut(); }
    @Benchmark public NDArray eagerOut() { return matrix.eagerOut(); }
    @TearDown public void close() { matrix.close(); }
}
