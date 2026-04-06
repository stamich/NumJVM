package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import org.openjdk.jmh.annotations.*;

/** Identical-view output benchmark; input reset is outside the measured method. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ExpressionInPlaceBenchmark {
    @Param({"256", "65536", "1048576"}) public int size;
    @Param({"C"}) public Layouts.Layout layout;
    @Param({"FLOAT32", "FLOAT64", "MIXED"}) public String types;
    @Param({"AUTO"}) public String strategy;
    private NDArray input, seed, other, temp1, temp2;
    private Jit.Context context;
    private Expression expression;

    @Setup(Level.Trial) public void setup() {
        DType inputType = types.equals("FLOAT32") ? DType.FLOAT32 : DType.FLOAT64;
        DType otherType = types.equals("FLOAT64") ? DType.FLOAT64 : DType.FLOAT32;
        input = Layouts.make(size, layout, inputType);
        seed = input.copy();
        other = Layouts.make(size, layout, otherType);
        temp1 = NJ.zeros(inputType, Layouts.rows(size), Layouts.cols(size));
        temp2 = NJ.zeros(inputType, Layouts.rows(size), Layouts.cols(size));
        if (!strategy.equals("EAGER")) {
            context = Jit.context(Jit.Mode.valueOf(strategy));
            expression = context.trace(2, i -> i.input(0).multiply(i.input(1)).add(i.input(1)).divide(i.input(1)));
        }
        NDArray expected = NJ.divide(NJ.add(NJ.multiply(seed, other), other), other);
        execute();
        for (long i = 0; i < Layouts.rows(size); i++) for (long j = 0; j < Layouts.cols(size); j++) {
            if (Double.doubleToLongBits(expected.getDouble(i, j)) != Double.doubleToLongBits(input.getDouble(i, j)))
                throw new AssertionError("in-place result mismatch");
        }
    }

    @Setup(Level.Invocation) public void reset() { Ufuncs.ADD.apply(seed, 0.0, input); }

    @Benchmark public NDArray execute() {
        if (strategy.equals("EAGER")) {
            Ufuncs.MULTIPLY.apply(input, other, temp1);
            Ufuncs.ADD.apply(temp1, other, temp2);
            return Ufuncs.DIVIDE.apply(temp2, other, input);
        }
        return expression.evalInto(input, input, other);
    }

    @TearDown(Level.Trial) public void close() { if (context != null) context.close(); }
}
