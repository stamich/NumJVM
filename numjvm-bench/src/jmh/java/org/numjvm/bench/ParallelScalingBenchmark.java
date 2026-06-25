package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import org.numjvm.ParallelContext;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Ufuncs;
import org.openjdk.jmh.annotations.*;

/** End-to-end warm execution, with a single JMH thread and explicit worker count. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ParallelScalingBenchmark {
    @Param({"1", "2", "4", "8", "16"}) public int workers;
    @Param({"65536", "1048576"}) public int size;
    @Param({"FLOAT32", "FLOAT64"}) public DType dtype;
    @Param({"ADD", "CHAIN", "SUM"}) public String operation;
    private ParallelContext pool;
    private Jit.Context jit;
    private Expression expression;
    private NDArray a, b, out, temporary;

    @Setup public void setup() {
        pool = NJ.parallel(workers);
        jit = Jit.context(Jit.Mode.AUTO);
        expression = jit.trace(2, i -> i.input(0).add(i.input(1)).square());
        a = NJ.ones(dtype, size); b = NJ.ones(dtype, size);
        out = NJ.zeros(dtype, size); temporary = NJ.zeros(dtype, size);
        NDArray actual = parallel();
        double expected = operation.equals("SUM") ? size : operation.equals("CHAIN") ? 4 : 2;
        if (actual.getDouble(operation.equals("SUM") ? new long[]{} : new long[]{size - 1}) != expected)
            throw new AssertionError("incorrect benchmark result");
        System.out.println("NUMJVM_PARALLEL_DECISION=" + pool.lastDecision());
        System.out.println("NUMJVM_PROVIDER=" + BackendInfo.simdProvider());
        System.out.println("NUMJVM_WORKERS=" + workers);
    }

    @Benchmark public NDArray parallel() {
        return switch (operation) {
            case "ADD" -> pool.apply(Ufuncs.ADD, a, b, out);
            case "CHAIN" -> expression.evalInto(pool, out, a, b);
            case "SUM" -> pool.sum(a);
            default -> throw new IllegalArgumentException(operation);
        };
    }
    @Benchmark public NDArray sequential() {
        return switch (operation) {
            case "ADD" -> Ufuncs.ADD.apply(a, b, out);
            case "CHAIN" -> Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(a, b, temporary), out);
            case "SUM" -> Reductions.sum(a);
            default -> throw new IllegalArgumentException(operation);
        };
    }
    @TearDown public void close() { pool.close(); jit.close(); }
}
