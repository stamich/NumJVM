package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import org.openjdk.jmh.annotations.*;

/** Compares warm expression execution against the equivalent eager operations. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ExpressionBenchmark {
    @Param({"256", "65536", "1048576"}) public int size;
    private NDArray a, b, c, d, output, temporary1, temporary2;
    private Jit.Context block, generated;
    private Expression blockPlan, generatedPlan;

    @Setup public void setup() {
        a = NJ.ones(DType.FLOAT64, size); b = NJ.ones(DType.FLOAT64, size);
        c = NJ.ones(DType.FLOAT64, size); d = NJ.ones(DType.FLOAT64, size);
        output = NJ.zeros(DType.FLOAT64, size);
        temporary1 = NJ.zeros(DType.FLOAT64, size); temporary2 = NJ.zeros(DType.FLOAT64, size);
        block = Jit.context(Jit.Mode.BLOCK); generated = Jit.context(Jit.Mode.REQUIRED);
        blockPlan = block.trace(4, i -> i.input(0).multiply(i.input(1)).add(i.input(2)).divide(i.input(3)));
        generatedPlan = generated.trace(4, i -> i.input(0).multiply(i.input(1)).add(i.input(2)).divide(i.input(3)));
        generatedPlan.evalInto(output, a, b, c, d);
    }

    @Benchmark public NDArray eagerAllocating() { return NJ.divide(NJ.add(NJ.multiply(a, b), c), d); }
    @Benchmark public NDArray eagerOut() {
        Ufuncs.MULTIPLY.apply(a, b, temporary1);
        Ufuncs.ADD.apply(temporary1, c, temporary2);
        return Ufuncs.DIVIDE.apply(temporary2, d, output);
    }
    @Benchmark public NDArray blockOut() { return blockPlan.evalInto(output, a, b, c, d); }
    @Benchmark public NDArray generatedOut() { return generatedPlan.evalInto(output, a, b, c, d); }
    @Benchmark public NDArray blockAllocating() { return blockPlan.eval(a, b, c, d); }
    @Benchmark public NDArray generatedAllocating() { return generatedPlan.eval(a, b, c, d); }
    @TearDown public void close() { block.close(); generated.close(); }
}
