package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.*;

/** Compares byte caps for operation-local block scratch. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ExpressionScratchBenchmark {
    @Param({"4096", "16384", "65536"}) public int scratchBytes;
    @Param({"65536", "1048576"}) public int size;
    private Jit.Context context;
    private Expression plan;
    private NDArray input, output;
    @Setup public void setup() {
        context = Jit.context(Jit.Mode.BLOCK, 64, scratchBytes);
        plan = context.trace(1, i -> i.input(0).square().add(1).sqrt());
        input = NJ.ones(size); output = NJ.zeros(size);
        plan.evalInto(output, input);
        if (output.getDouble(0) != Math.sqrt(2)) throw new AssertionError("incorrect result");
    }
    @Benchmark public NDArray execute() { return plan.evalInto(output, input); }
    @TearDown public void close() { context.close(); }
}
